"""H11: replay with an asynchronous router, and hibernation only for processes idle for a while.

One thread per model. Swaps and restores run in parallel, because a restore does not stall other
contexts (H10). Kernels run one request at a time. HIB_AFTER sets the idle seconds before a parked
process is checkpointed; a negative value turns hibernation off.
"""
import json, multiprocessing as mp, os, queue, subprocess, sys, threading, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vf
from h7 import FIT, NMODELS, SOCK, TOOL, trace, worker
rate, secs = float(sys.argv[1]), float(sys.argv[2]); AFTER = float(os.environ.get("HIB_AFTER", -1))

def used():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])

def main():
    ctx = mp.get_context("spawn"); br = vf.Client(SOCK); pipes, procs, ref = [], [], []
    for i in range(NMODELS):
        a, b = ctx.Pipe(); p = ctx.Process(target=worker, args=(i, b), daemon=True); p.start()
        pipes.append(a); procs.append(p); ref.append(a.recv()); br.evict(f"m{i}")
    cond = threading.Condition(); gpu = threading.Lock(); resident = {}; pins = [0] * NMODELS; loading = set(); hib = {}; last = [0.0] * NMODELS
    stats = dict(misses=0, wrong=0, restores=0, swap=0.0, restore=0.0, busy=0.0)
    def call(i, op):
        pipes[i].send(op); return pipes[i].recv()
    def toggle(i):
        subprocess.run([TOOL, "--toggle", "--pid", str(procs[i].pid)], check=True, capture_output=True)
    def acquire(i):
        with cond:
            if i in resident:
                resident[i] = resident.pop(i); pins[i] += 1; return
            stats["misses"] += 1; t = time.perf_counter()
            while len(resident) + len(loading) >= FIT:
                v = next((v for v in resident if pins[v] == 0), None)
                if v is None:
                    cond.wait(); continue
                resident.pop(v); call(v, "park"); br.evict(f"m{v}"); last[v] = time.perf_counter()
            loading.add(i); th = hib.pop(i, None)
        if th:
            th.join(); r = time.perf_counter(); toggle(i); r = time.perf_counter() - r; stats["restores"] += 1; stats["restore"] += r
        call(i, "resume")
        with cond:
            loading.discard(i); resident[i] = 1; pins[i] += 1; stats["swap"] += time.perf_counter() - t; cond.notify_all()
    def serve(i):
        acquire(i)
        with gpu:
            val, d = call(i, "infer")
        stats["busy"] += d; stats["wrong"] += val != ref[i]
        with cond:
            pins[i] -= 1; last[i] = time.perf_counter(); cond.notify_all()
    def hibernator():
        while True:
            time.sleep(0.2); now = time.perf_counter()
            with cond:
                todo = [i for i in range(NMODELS) if i not in resident and i not in loading and i not in hib and now - last[i] >= AFTER]
                for i in todo:
                    th = threading.Thread(target=toggle, args=(i,)); th.start(); hib[i] = th
    if AFTER >= 0:
        threading.Thread(target=hibernator, daemon=True).start()
    time.sleep(8); idle = used()
    for k in stats:
        stats[k] = 0
    tr = trace(); qs = [queue.Queue() for _ in range(NMODELS)]; lats, hit, miss = [], [], []; rec = threading.Lock(); done = threading.Event(); start = time.perf_counter()
    samples = []
    def sampler():
        while not done.is_set():
            samples.append(used()); time.sleep(1)
    threading.Thread(target=sampler, daemon=True).start()
    def model_loop(i):
        while True:
            at = qs[i].get(); was = i in resident; serve(i); l = time.perf_counter() - start - at
            with rec:
                lats.append(l); (hit if was else miss).append(l)
                if len(lats) == len(tr):
                    done.set()
    for i in range(NMODELS):
        threading.Thread(target=model_loop, args=(i,), daemon=True).start()
    for at, i in tr:
        d = start + at - time.perf_counter()
        if d > 0:
            time.sleep(d)
        qs[i].put(at)
    done.wait(); wall = time.perf_counter() - start; lats.sort()
    pct = lambda a, p: round(sorted(a)[min(len(a) - 1, int(len(a) * p))] * 1e3, 1) if a else None
    print(json.dumps(dict(impl="vf_async", hib_after_s=AFTER, n=len(lats), misses=stats["misses"], wrong=stats["wrong"], restores=stats["restores"], restore_ms_mean=round(stats["restore"] / max(stats["restores"], 1) * 1e3, 1),
        swap_ms_mean=round(stats["swap"] / max(stats["misses"], 1) * 1e3, 1), infer_frac=round(stats["busy"] / wall, 3), p50_ms=pct(lats, .5), p99_ms=pct(lats, .99), max_ms=round(lats[-1] * 1e3, 1),
        miss_p50_ms=pct(miss, .5), miss_p99_ms=pct(miss, .99), hit_p99_ms=pct(hit, .99), used_idle_mib=idle, used_mean_mib=round(sum(samples) / len(samples)), used_max_mib=max(samples))), flush=True)
    os._exit(0)

if __name__ == "__main__":
    main()
