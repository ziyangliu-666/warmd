"""H10: open questions after the first replays.

  diag    why ResNet and Whisper outputs differ on mapped weights; compare against a plain clone
  stall   does a checkpoint or restore of one process stall kernels in another?
  pss     real host memory per process, spawned against forked from a zygote
  lease   eight native contexts sharing a GPU under a mutex lease instead of driver time slicing
"""
import json, multiprocessing as mp, os, struct, subprocess, sys, threading, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vf
from common import path
from h8 import build, forward, zoo_cases
SOCK = "/tmp/vf.sock"; TOOL = "/workspace/exp/cuda-checkpoint"
def emit(**k): print(json.dumps(k), flush=True)

def write_aligned(tensors, f, align):
    """Write a safetensors file whose payload starts each tensor at `align` bytes from the file start."""
    names = list(tensors); hdr = {}; pos = 0
    for k in names:
        pos = (pos + align - 1) & ~(align - 1); n = tensors[k].numel() * tensors[k].element_size()
        hdr[k] = dict(dtype="F32", shape=list(tensors[k].shape), data_offsets=[pos, pos + n]); pos += n
    h = json.dumps(hdr).encode(); start = 8 + len(h); pad = (-start) % align; h += b" " * pad
    with open(f, "wb") as o:
        o.write(struct.pack("<Q", len(h))); o.write(h); base = o.tell()
        for k in names:
            o.seek(base + hdr[k]["data_offsets"][0]); o.write(tensors[k].cpu().numpy().tobytes())

def diag_child(name, q):
    try:
        make, run, _ = zoo_cases()[name]; torch.manual_seed(0); m = make().eval(); vf.init()
        names = dict(m.named_parameters(remove_duplicate=False)); first = {}; alias = {}
        for k, p in names.items():
            alias[k] = first.setdefault(id(p), k)
        cpu = {k: names[k].detach().clone() for k in first.values()}
        m = m.to("cuda")
        def infer():
            with torch.inference_mode():
                return run(m).float().cpu()
        def install(sd):
            for k in names:
                mod, _, leaf = k.rpartition("."); m.get_submodule(mod)._parameters[leaf] = torch.nn.Parameter(sd[alias[k]], requires_grad=False)
        ref = infer(); out = dict(model=name, ref_abs_mean=float(ref.abs().mean()))
        def report(label, sd):
            install(sd); o = infer(); same = all(bool((sd[k].cpu() == cpu[k]).all()) for k in cpu)
            out[label] = dict(max_abs_diff=float((o - ref).abs().max()), weights_equal=same)
        report("same_weights_new_allocation", {k: v.cuda().clone() for k, v in cpu.items()})
        report("again_new_allocation", {k: v.cuda().clone() for k, v in cpu.items()})
        install({k: v.cuda() for k, v in cpu.items()}); ref2 = infer(); out["reinstalled_vs_ref"] = float((ref2 - ref).abs().max())
        c = vf.Client(SOCK)
        for align in (8, 64, 4096):
            f = f"/tmp/diag_{name}_{align}.safetensors"; write_aligned(cpu, f, align); report(f"mapped_align_{align}", c.open(f"d{name}{align}", f))
        q.put(out)
    except Exception as e:
        q.put(dict(model=name, exception=str(e)[:300]))

def t_diag():
    ctx = mp.get_context("spawn")
    for name in ("resnet_batchnorm", "whisper", "bert"):
        q = ctx.Queue(); p = ctx.Process(target=diag_child, args=(name, q)); p.start(); emit(test="diag", **q.get()); p.join()

def bg_loop(ready, stop, q):
    a = torch.randn(1024, 1024, device="cuda"); float((a @ a).sum()); ready.set(); lat = []
    while not stop.is_set():
        t = time.perf_counter(); float((a @ a).sum()); lat.append((time.perf_counter(), time.perf_counter() - t))
    q.put(lat)

def idle_child(conn):
    torch.zeros(1, device="cuda"); torch.cuda.synchronize(); a = torch.randn(64, 64, device="cuda"); float((a @ a).sum()); conn.send(1); conn.recv()

def t_stall():
    ctx = mp.get_context("spawn"); r, s, q = ctx.Event(), ctx.Event(), ctx.Queue(); bg = ctx.Process(target=bg_loop, args=(r, s, q)); bg.start(); r.wait()
    a, b = ctx.Pipe(); v = ctx.Process(target=idle_child, args=(b,)); v.start(); a.recv(); time.sleep(3); marks = []
    for _ in range(6):
        t0 = time.perf_counter(); subprocess.run([TOOL, "--toggle", "--pid", str(v.pid)], check=True, capture_output=True); marks.append((t0, time.perf_counter())); time.sleep(1)
    s.set(); lat = q.get(); bg.join(); a.send(1); v.join()
    inside = [d for t, d in lat if any(x <= t <= y + d for x, y in marks)]; outside = [d for t, d in lat if not any(x <= t <= y + d for x, y in marks)]
    p = lambda v, f: round(sorted(v)[min(len(v) - 1, int(len(v) * f))] * 1e3, 2) if v else None
    emit(test="stall", toggle_ms=[round((y - x) * 1e3) for x, y in marks], outside_p50=p(outside, .5), outside_p99=p(outside, .99), outside_max=p(outside, 1), inside_n=len(inside), inside_p50=p(inside, .5), inside_p99=p(inside, .99), inside_max=p(inside, 1))

def pss_mib(pid):
    for l in open(f"/proc/{pid}/smaps_rollup"):
        if l.startswith("Pss:"):
            return int(l.split()[1]) / 1024

def pss_child(conn):
    from transformers import LlamaForCausalLM
    vf.init(); c = vf.Client(SOCK); sd = c.open("pss", path(2)); m = build(sd, 2); forward(m); torch.cuda.synchronize(); c.park("pss"); conn.send(1); conn.recv()

def t_pss():
    ctx = mp.get_context("spawn"); N = 8; pipes = []; procs = []
    for _ in range(N):
        a, b = ctx.Pipe(); p = ctx.Process(target=pss_child, args=(b,), daemon=True); p.start(); a.recv(); pipes.append(a); procs.append(p)
    time.sleep(1); spawned = sum(pss_mib(p.pid) for p in procs) / N
    for p in procs:
        subprocess.run([TOOL, "--toggle", "--pid", str(p.pid)], check=True, capture_output=True)
    hib = sum(pss_mib(p.pid) for p in procs) / N
    for p in procs:
        subprocess.run([TOOL, "--toggle", "--pid", str(p.pid)], check=True, capture_output=True)
    for a in pipes:
        a.send(1)
    [p.join() for p in procs]
    from transformers import LlamaForCausalLM
    fctx = mp.get_context("fork"); pipes = []; procs = []
    for _ in range(N):
        a, b = fctx.Pipe(); p = fctx.Process(target=pss_child, args=(b,), daemon=True); p.start(); a.recv(); pipes.append(a); procs.append(p)
    time.sleep(1); forked = sum(pss_mib(p.pid) for p in procs) / N
    for a in pipes:
        a.send(1)
    [p.join() for p in procs]
    emit(test="pss", procs=N, spawned_parked_mib=round(spawned), spawned_hibernated_mib=round(hib), zygote_forked_parked_mib=round(forked))

def lease_child(i, n, secs, bar, res, sems, hold):
    a = torch.randn(n, n, device="cuda"); float((a @ a).sum()); lat = []; leases = 0; bar.wait(); end = time.perf_counter() + secs
    if sems is None:
        while time.perf_counter() < end:
            t = time.perf_counter(); float((a @ a).sum()); lat.append(time.perf_counter() - t)
        res.put((lat, 0)); return
    # Strict round robin: an mp.Lock is unfair, and one process kept it in the first attempt.
    queued = time.perf_counter()
    while time.perf_counter() < end:
        if not sems[i].acquire(timeout=0.05):
            continue
        leases += 1; stop = time.perf_counter() + hold
        while True:
            float((a @ a).sum()); now = time.perf_counter(); lat.append(now - queued); queued = now
            if now >= stop:
                break
        sems[(i + 1) % len(sems)].release(); queued = time.perf_counter()
    res.put((lat, leases))

def t_lease():
    ctx = mp.get_context("spawn"); N = 8
    for n, secs in ((256, 6), (4096, 8)):
        for hold in (None, 0.0002, 0.0005, 0.002, 0.01, 0.05):
            bar, res = ctx.Barrier(N), ctx.Queue(); sems = None if hold is None else [ctx.Semaphore(1 if i == 0 else 0) for i in range(N)]
            ps = [ctx.Process(target=lease_child, args=(i, n, secs, bar, res, sems, hold or 0)) for i in range(N)]
            [p.start() for p in ps]; rs = [res.get() for _ in ps]; [p.join() for p in ps]
            per = [len(r[0]) / secs for r in rs]; lat = sorted(sum((r[0] for r in rs), []))
            emit(test="fair_lease", n=n, procs=N, hold_ms=None if hold is None else hold * 1e3, ops_s=round(sum(per), 1), min_proc_ops_s=round(min(per), 1), max_proc_ops_s=round(max(per), 1), leases_s=round(sum(r[1] for r in rs) / secs, 1), p50_ms=round(lat[len(lat) // 2] * 1e3, 2), p99_ms=round(lat[int(len(lat) * .99)] * 1e3, 2))

if __name__ == "__main__":
    {"diag": t_diag, "stall": t_stall, "pss": t_pss, "lease": t_lease}[sys.argv[1]]()
