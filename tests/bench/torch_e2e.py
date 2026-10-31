# torch_e2e.py - end-to-end throughput/latency of many short *unmodified* PyTorch tasks, each in
# its own process. Modes:
#   exec   : fresh `python3 torch_task.py` per task (pays import torch every time)
#   zygote : this process imports torch (NO CUDA init), then fork()s one child per task
#   warm   : (shim only) parent ALSO initializes CUDA + cuBLAS, then forks. Natively impossible: a forked
#            child cannot use a parent's CUDA context. With the shim no CUDA state lives in the process.
#            Each task uses its own seed; results are checked against a CPU reference.
# Run it once natively and once with LD_LIBRARY_PATH=build/lib (through warmd).
#   python3 torch_e2e.py <exec|zygote> <ntasks> <concurrency> [label]
import json, os, sys, time, subprocess
mode, n, conc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
label = sys.argv[4] if len(sys.argv) > 4 else ""
HERE = os.path.dirname(os.path.abspath(__file__))
REF = {"add": -523.3961181640625, "relu": 0.7937963604927063, "mm": 832876.375,
       "softmax": 0.017304101958870888, "cpu": 127.62001509445417}

def check(out):
    return all(abs(out.get(k, 1e30) - v) <= 1e-6 * max(1.0, abs(v)) for k, v in REF.items())

lat, bad, forks, tasks = [], 0, [], []
t0 = time.perf_counter()
if mode == "exec":
    running = []
    i = 0
    while i < n or running:
        while i < n and len(running) < conc:
            running.append((time.perf_counter(), subprocess.Popen([sys.executable, f"{HERE}/torch_task.py"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL))); i += 1
        ts, p = running.pop(0); o, _ = p.communicate()
        lat.append(time.perf_counter() - ts)
        try: bad += not check(json.loads(o))
        except Exception: bad += 1
else:
    import torch  # T1 paid once here
    torch.set_num_threads(1)  # an OpenMP pool used before fork() deadlocks the children
    def inputs(seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(256, 256, generator=g), torch.randn(256, 256, generator=g)
    def ref(seed):
        a, b = inputs(seed)
        return {"add": float((a + b).sum()), "relu": float((torch.relu(a) * 2.0).mean()),
                "mm": float((a @ b).abs().sum()), "softmax": float(torch.softmax(a, dim=1)[0, :8].sum()),
                "cpu": float((a * b).double().sum())}
    REFS = [ref(i) for i in range(n)]
    def check_i(out, i):
        return all(abs(out.get(k, 1e30) - v) <= 1e-3 * max(1.0, abs(v)) for k, v in REFS[i].items())
    def task(seed=0):
        a, b = inputs(seed)
        x = a.cuda(); y = b.cuda(); out = {}
        out["add"] = float((x + y).sum().item())
        out["relu"] = float((torch.relu(x) * 2.0).mean().item())
        out["mm"] = float((x @ y).abs().sum().item())
        out["softmax"] = float(torch.softmax(x, dim=1)[0, :8].sum().item())
        out["cpu"] = float((x * y).cpu().double().sum())
        torch.cuda.synchronize()
        return out
    if mode == "warm":
        task(0)                                   # warm cudart, torch CUDA state, cuBLAS handle, modules
        torch.cuda.synchronize()
        torch._C._cuda_clearCublasWorkspaces()    # children must not share the parent's workspace
        torch.cuda.empty_cache()                  # ...or the parent's cached allocator blocks
    else:
        assert not torch.cuda.is_initialized()
    t0 = time.perf_counter()
    running = {}
    i = 0
    while i < n or running:
        while i < n and len(running) < conc:
            r, w = os.pipe(); ts = time.perf_counter(); pid = os.fork()
            if pid: forks.append(time.perf_counter() - ts)
            if pid == 0:
                os.close(r)
                if mode == "warm":   # torch's guard is right for native CUDA; with the shim the child owns no CUDA state
                    torch.cuda._is_in_bad_fork = lambda: False
                tt = time.perf_counter()
                try: o = task(i); o["task_ms"] = (time.perf_counter() - tt) * 1e3; res = json.dumps(o)
                except Exception as e: res = json.dumps({"err": str(e)})
                os.write(w, res.encode()); os._exit(0)
            os.close(w); running[pid] = (ts, r, i); i += 1
        pid, _ = os.wait()
        ts, r, ti = running.pop(pid)
        data = b""
        while True:
            chunk = os.read(r, 65536)
            if not chunk: break
            data += chunk
        os.close(r)
        lat.append(time.perf_counter() - ts)
        try: d = json.loads(data); tasks.append(d.get("task_ms", 0)); ok = check_i(d, ti)
        except Exception: ok = False
        if not ok:
            bad += 1
            if os.environ.get("E2E_SHOW") and bad <= 2: print("BAD", ti, data[:300], REFS[ti], file=sys.stderr)
wall = time.perf_counter() - t0
lat.sort()
print(json.dumps({"label": label, "mode": mode, "n": n, "conc": conc, "wall_s": round(wall, 2),
                  "tasks_per_s": round(n / wall, 2), "p50_ms": round(lat[len(lat) // 2] * 1e3, 1),
                  "p99_ms": round(lat[min(len(lat) - 1, int(len(lat) * 0.99))] * 1e3, 1), "wrong": bad,
                  "fork_ms": round(sum(forks) / len(forks) * 1e3, 2) if forks else None,
                  "task_ms_p50": round(sorted(tasks)[len(tasks) // 2], 1) if tasks else None}))
