"""H3: compare one context per process (native) with warmd's single shared context.

Modes:
  overhead      device memory used per native context, for 1, 2, 4, 8 processes
  ipc           share an allocation across native contexts: import cost, memory delta, write access
  fault         one native context faults; do the other contexts survive?
  wild          a second process writes to the first process's device address
  conc          aggregate throughput and latency for N processes running matmuls
"""
import json, multiprocessing as mp, os, subprocess, sys, time
import torch
import triton, triton.language as tl

def emit(**kw): print(json.dumps(kw), flush=True)
def used_mib():
    o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    return int(o.split()[0])

def make_wild():
    @triton.jit
    def wild(p, off, val):
        tl.store(p + off, val)
    return wild

def holder(ready, stop):
    torch.zeros(1, device="cuda"); torch.cuda.synchronize(); ready.set(); stop.wait()

def overhead(ctx):
    base = used_mib()
    for n in (1, 2, 4, 8):
        stop = ctx.Event(); evs = [ctx.Event() for _ in range(n)]
        ps = [ctx.Process(target=holder, args=(e, stop)) for e in evs]
        [p.start() for p in ps]; [e.wait() for e in evs]; time.sleep(2)
        u = used_mib(); stop.set(); [p.join() for p in ps]; time.sleep(2)
        emit(test="overhead", procs=n, used_mib=u, per_proc_mib=round((u - base) / n, 1))

def ipc_child(q, res, go):
    t = time.perf_counter(); y = q.get(); t = time.perf_counter() - t
    used = used_mib(); s = float(y[:1000].sum()); y[0] = 7.0; torch.cuda.synchronize()
    res.put(dict(import_ms=round(t * 1e3, 2), used_after_import=used, read_ok=s == 1000.0))
    go.wait()

def ipc(ctx):
    x = torch.ones(2_000_000_000, device="cuda", dtype=torch.float32); torch.cuda.synchronize()
    used = used_mib(); q, res, go = ctx.Queue(), ctx.Queue(), ctx.Event()
    p = ctx.Process(target=ipc_child, args=(q, res, go)); p.start(); q.put(x)
    r = res.get(); time.sleep(1); after = used_mib(); seen = float(x[0])
    emit(test="ipc", size_gb=8, used_before=used, used_with_importer=after, importer_writes_visible=seen == 7.0, **r)
    go.set(); p.join()

def fault_child(q, go, res):
    y = q.get(); s = float(y[:10].sum()); wild = make_wild(); go.wait()
    z = torch.zeros(16, device="cuda")
    try:
        wild[(1,)](z, 1 << 38, 1.0); torch.cuda.synchronize(); out = "no_fault"
    except Exception as e:
        out = str(e)[:90]
    res.put(out)

def bystander(res, go, done):
    a = torch.ones(1000, 1000, device="cuda"); float((a @ a).sum()); go.wait(); done.wait()
    try:
        res.put(("bystander_ok", float((a @ a).sum()) > 0))
    except Exception as e:
        res.put(("bystander_err", str(e)[:90]))

def fault(ctx):
    x = torch.ones(1000, device="cuda"); q, res, go, done = ctx.Queue(), ctx.Queue(), ctx.Event(), ctx.Event()
    bres = ctx.Queue(); bgo = ctx.Event()
    f = ctx.Process(target=fault_child, args=(q, go, res)); b = ctx.Process(target=bystander, args=(bres, bgo, done))
    f.start(); b.start(); q.put(x); go.set(); bgo.set()
    t = time.perf_counter(); out = res.get(); t = time.perf_counter() - t; f.join(); done.set()
    br = bres.get(); b.join()
    emit(test="fault", culprit=out, report_ms=round(t * 1e3), bystander=br, owner_ok=float(x.sum()) == 1000.0)

def wild_child(addr, res):
    wild = make_wild(); z = torch.zeros(16, device="cuda", dtype=torch.float32)
    try:
        wild[(1,)](z, (addr - z.data_ptr()) // 4, 9.0); torch.cuda.synchronize(); res.put("store_completed")
    except Exception as e:
        res.put("faulted: " + str(e)[:70])

def wild(ctx, impl):
    x = torch.zeros(1024, device="cuda", dtype=torch.float32); torch.cuda.synchronize(); res = ctx.Queue()
    p = ctx.Process(target=wild_child, args=(x.data_ptr(), res)); p.start(); out = res.get(); p.join()
    try:
        v = float(x[0]); state = "alive"
    except Exception as e:
        v = None; state = str(e)[:70]
    emit(test="wild", impl=impl, attacker=out, victim_value=v, victim=state)

def conc_child(n, secs, bar, res):
    a = torch.randn(n, n, device="cuda"); float((a @ a).sum()); lat = []
    bar.wait(); end = time.perf_counter() + secs
    while time.perf_counter() < end:
        t = time.perf_counter(); float((a @ a).sum()); lat.append(time.perf_counter() - t)
    res.put(lat)

def conc(ctx, impl):
    for n, secs in ((256, 6), (4096, 10)):
        for procs in (1, 2, 4, 8):
            bar, res = ctx.Barrier(procs), ctx.Queue()
            ps = [ctx.Process(target=conc_child, args=(n, secs, bar, res)) for _ in range(procs)]
            [p.start() for p in ps]; lat = sorted(sum((res.get() for _ in ps), [])); [p.join() for p in ps]
            emit(test="conc", impl=impl, n=n, procs=procs, ops_s=round(len(lat) / secs, 1),
                 p50_ms=round(lat[len(lat) // 2] * 1e3, 2), p99_ms=round(lat[int(len(lat) * .99)] * 1e3, 2))

if __name__ == "__main__":
    ctx = mp.get_context("spawn"); mode, impl = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "native"
    {"overhead": lambda: overhead(ctx), "ipc": lambda: ipc(ctx), "fault": lambda: fault(ctx),
     "wild": lambda: wild(ctx, impl), "conc": lambda: conc(ctx, impl)}[mode]()
