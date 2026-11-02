"""H7: replay the H4 trace with one process per model on device-memory files.

Every model keeps its own process, context and Python state, so tenants stay isolated. A router
keeps five files resident. A miss parks the least recently used process, evicts its file, and
resumes the requested process. Requests run one at a time, as in H4.
"""
import json, multiprocessing as mp, os, queue, random, subprocess, sys, threading, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vf
from common import config, path
SOCK = "/tmp/vf.sock"; NMODELS, FIT, GB = 20, 5, 2
HIBERNATE = os.environ.get("HIBERNATE") == "1"; TOOL = "/workspace/exp/cuda-checkpoint"
rate, secs = float(sys.argv[1]), float(sys.argv[2])

def trace():
    # Same generator and seed as h4.py, so both runs replay identical requests.
    rng = random.Random(7); out = []
    weights = [1 / (i + 1) for i in range(NMODELS)]; total = sum(weights)
    for i, w in enumerate(weights):
        r = rate * w / total; t = rng.expovariate(0.1)
        while t < secs:
            burst = rng.expovariate(1 / 10); on_rate = r * 4; end = min(t + burst, secs)
            while t < end:
                out.append((t, i)); t += rng.expovariate(on_rate)
            t = end + rng.expovariate(1 / 30)
    return sorted(out)

def worker(i, conn):
    from accelerate import init_empty_weights
    from transformers import LlamaForCausalLM
    vf.init(); c = vf.Client(SOCK); name = f"m{i}"; sd = c.open(name, path(GB))
    with init_empty_weights():
        m = LlamaForCausalLM(config(GB))
    m.load_state_dict(sd, assign=True); m.eval(); m.model.rotary_emb.to("cuda")
    ids = torch.arange(512, device="cuda").unsqueeze(0)
    def run():
        with torch.inference_mode():
            for _ in range(2):
                out = m(ids).logits
            v = float(out.float().sum())
        return v
    ref = run(); c.park(name); torch.cuda.empty_cache(); conn.send(ref)
    while True:
        op = conn.recv()
        if op == "stop":
            return
        t = time.perf_counter()
        if op == "park":
            c.park(name)
        elif op == "resume":
            c.resume(name, path(GB))
        elif op == "infer":
            conn.send((run(), time.perf_counter() - t)); continue
        conn.send(time.perf_counter() - t)

def main():
    ctx = mp.get_context("spawn"); br = vf.Client(SOCK); pipes, procs, ref = [], [], []
    for i in range(NMODELS):
        a, b = ctx.Pipe(); p = ctx.Process(target=worker, args=(i, b), daemon=True); p.start()
        pipes.append(a); procs.append(p); ref.append(a.recv()); br.evict(f"m{i}")
    idle = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
    resident = {}; swap = busy = 0.0; misses = wrong = 0; hib = {}; restore = []
    def toggle(i):
        subprocess.run([TOOL, "--toggle", "--pid", str(procs[i].pid)], check=True, capture_output=True)
    def hibernate(i):
        # A checkpoint only releases memory, so it runs in the background and no request waits for it.
        th = threading.Thread(target=toggle, args=(i,)); th.start(); hib[i] = th
    def wake(i):
        if i in hib:
            hib.pop(i).join(); t = time.perf_counter(); toggle(i); restore.append(time.perf_counter() - t)
    def call(i, op):
        pipes[i].send(op); return pipes[i].recv()
    if HIBERNATE:
        for i in range(NMODELS):
            hibernate(i)
        [hib[i].join() for i in hib]; time.sleep(1)
        idle = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
    def serve(i):
        nonlocal swap, busy, misses, wrong
        if i in resident:
            resident[i] = resident.pop(i)
        else:
            t = time.perf_counter()
            if len(resident) >= FIT:
                v = next(iter(resident)); resident.pop(v); call(v, "park"); br.evict(f"m{v}")
                if HIBERNATE:
                    hibernate(v)
            wake(i); call(i, "resume"); resident[i] = 1; swap += time.perf_counter() - t; misses += 1
        val, d = call(i, "infer"); busy += d; wrong += val != ref[i]
    for i in range(FIT):
        serve(i)
    serve(0); t = time.perf_counter()
    for _ in range(5):
        serve(0)
    svc = (time.perf_counter() - t) / 5; swap = busy = 0.0; misses = wrong = 0
    tr = trace(); q = queue.Queue(); start = time.perf_counter()
    def feeder():
        for at, i in tr:
            d = start + at - time.perf_counter()
            if d > 0:
                time.sleep(d)
            q.put((at, i))
        q.put(None)
    threading.Thread(target=feeder, daemon=True).start(); lats, hit, miss = [], [], []
    while (item := q.get()) is not None:
        at, i = item; was = i in resident; serve(i); l = time.perf_counter() - start - at; lats.append(l); (hit if was else miss).append(l)
    wall = time.perf_counter() - start; lats.sort()
    pct = lambda a, p: round(sorted(a)[min(len(a) - 1, int(len(a) * p))] * 1e3, 1) if a else None
    used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
    print(json.dumps(dict(impl="vf_hibernate" if HIBERNATE else "vf_process_per_model", restore_ms_mean=round(sum(restore) / len(restore) * 1e3, 1) if restore else None, rate=rate, secs=secs, n=len(lats), service_ms=round(svc * 1e3, 1), misses=misses, wrong=wrong,
        miss_rate=round(misses / len(lats), 3), swap_ms_mean=round(swap / max(misses, 1) * 1e3, 1), swap_frac=round(swap / wall, 3), infer_frac=round(busy / wall, 3),
        p50_ms=pct(lats, .5), p99_ms=pct(lats, .99), max_ms=round(lats[-1] * 1e3, 1), miss_p50_ms=pct(miss, .5), miss_p99_ms=pct(miss, .99), hit_p99_ms=pct(hit, .99),
        used_idle_all_evicted_mib=idle, used_end_mib=used)), flush=True)
    os._exit(0)

if __name__ == "__main__":
    main()
