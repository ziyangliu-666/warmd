"""H28: context overhead with PyTorch, with and without NVIDIA MPS. Run in a container.

  mem     device memory for 1, 2, 4, 8 processes, after a first kernel
  conc    eight processes running matmuls, as in H3
  wild    a process wraps another process's device address and writes to it
"""
import json, multiprocessing as mp, os, subprocess, sys, time
import torch
sys.path.insert(0, "/workspace/exp")
mode, label = sys.argv[1], sys.argv[2]
def smi():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
def emit(**k): print(json.dumps(dict(label=label, test=mode, **k)), flush=True)
def holder(ready, stop):
    a = torch.randn(64, 64, device="cuda"); float((a @ a).sum()); del a; torch.cuda.empty_cache(); ready.set(); stop.wait()
def mem(ctx):
    base = smi(); out = []
    for n in (1, 2, 4, 8):
        stop = ctx.Event(); ev = [ctx.Event() for _ in range(n)]; ps = [ctx.Process(target=holder, args=(e, stop)) for e in ev]; t = time.perf_counter(); [p.start() for p in ps]; [e.wait() for e in ev]; w = time.perf_counter() - t; time.sleep(2); u = smi(); stop.set(); [p.join() for p in ps]; time.sleep(2)
        out.append(dict(procs=n, used_mib=u, per_proc_mib=round((u - base) / n, 1), start_s=round(w, 2)))
    emit(base_mib=base, runs=out)
def conc_child(n, secs, bar, res):
    a = torch.randn(n, n, device="cuda"); float((a @ a).sum()); lat = []; bar.wait(); end = time.perf_counter() + secs
    while time.perf_counter() < end:
        t = time.perf_counter(); float((a @ a).sum()); lat.append(time.perf_counter() - t)
    res.put(lat)
def conc(ctx):
    for n, secs in ((256, 6), (2048, 6)):
        for procs in (1, 8):
            bar, res = ctx.Barrier(procs), ctx.Queue(); ps = [ctx.Process(target=conc_child, args=(n, secs, bar, res)) for _ in range(procs)]; [p.start() for p in ps]; lat = sorted(sum((res.get() for _ in ps), [])); [p.join() for p in ps]
            emit(n=n, procs=procs, ops_s=round(len(lat) / secs, 1), p50_ms=round(lat[len(lat) // 2] * 1e3, 2), p99_ms=round(lat[int(len(lat) * .99)] * 1e3, 2))
def attacker(addr, q):
    import vf
    z = torch.zeros(4, device="cuda"); z.fill_(1.0); torch.cuda.synchronize(); ok = float(z[0]) == 1.0; t = vf.tensor_at(addr, [1024], 2, 32)
    try:
        t.fill_(9.0); torch.cuda.synchronize(); q.put(dict(control=ok, attacker="store_completed"))
    except Exception as e:
        q.put(dict(control=ok, attacker="faulted: " + str(e).split("\n")[0][:60]))
def by(q, go, done):
    a = torch.ones(512, 512, device="cuda"); r = float((a @ a).sum()); go.set(); done.wait()
    try:
        q.put(dict(bystander_same=float((a @ a).sum()) == r))
    except Exception as e:
        q.put(dict(bystander="died: " + str(e).split("\n")[0][:60]))
def wild(ctx):
    x = torch.zeros(1024, device="cuda", dtype=torch.float32); torch.cuda.synchronize(); q, q2, go, done = ctx.Queue(), ctx.Queue(), ctx.Event(), ctx.Event()
    b = ctx.Process(target=by, args=(q2, go, done)); b.start(); go.wait(); a = ctx.Process(target=attacker, args=(x.data_ptr(), q)); a.start(); r = q.get(); a.join(); time.sleep(1); done.set(); r.update(q2.get()); b.join()
    try:
        r.update(victim_value=float(x[0]), victim="alive")
    except Exception as e:
        r.update(victim="died: " + str(e).split("\n")[0][:60])
    emit(**r)
if __name__ == "__main__":
    {"mem": mem, "conc": conc, "wild": wild}[mode](mp.get_context("spawn")); os._exit(0)
