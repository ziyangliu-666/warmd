"""H4: replay a synthetic long-tail trace against an LRU device-memory cache in one process.

Twenty 2 GB models share a 10 GB budget, so five fit at once. Host copies stay in pinned memory.
A miss frees the least recently used model and copies the requested one from host memory. Kernels
stay loaded, so a miss pays no warm-up. The trace is synthetic: per-model on/off bursts with Zipf rates.
"""
import json, os, random, sys, threading, time, queue
import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from transformers import LlamaForCausalLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import config, path
impl, rate, secs = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
NMODELS, FIT, GB = 20, 5, 2

def trace():
    rng = random.Random(7); out = []
    weights = [1 / (i + 1) for i in range(NMODELS)]; total = sum(weights)
    for i, w in enumerate(weights):
        r = rate * w / total; t = rng.expovariate(0.1)
        while t < secs:
            burst = rng.expovariate(1 / 10); on_rate = r * 4; end = min(t + burst, secs)   # on 25% of the time
            while t < end:
                out.append((t, i)); t += rng.expovariate(on_rate)
            t = end + rng.expovariate(1 / 30)
    return sorted(out)

base = load_file(path(GB), device="cpu")
host = [{k: (v.clone() if v.is_floating_point() else v).pin_memory() for k, v in base.items()} for _ in range(NMODELS)]
del base
with init_empty_weights():
    shell = LlamaForCausalLM(config(GB))
ids = None
resident = {}   # model -> module on the GPU, kept in LRU order
swap_s = 0.0; misses = 0; busy = 0.0

def load(i):
    global swap_s, misses
    t = time.perf_counter()
    if len(resident) >= FIT:
        resident.pop(next(iter(resident))); torch.cuda.empty_cache()
    sd = {k: v.to("cuda", non_blocking=True) for k, v in host[i].items()}; torch.cuda.synchronize()
    with init_empty_weights():
        m = LlamaForCausalLM(config(GB))
    m.load_state_dict(sd, assign=True); m.eval(); m.model.rotary_emb.to("cuda")
    resident[i] = m; swap_s += time.perf_counter() - t; misses += 1

def serve(i):
    global busy
    if i in resident:
        resident[i] = resident.pop(i)
    else:
        load(i)
    t = time.perf_counter()
    with torch.inference_mode():
        for _ in range(2):
            resident[i](ids)
    torch.cuda.synchronize(); busy += time.perf_counter() - t

ids = torch.randint(0, 32000, (1, 512), device="cuda")
load(0); serve(0); t = time.perf_counter()   # The first pass includes kernel load, so it is not a service time.
for _ in range(5): serve(0)
svc = (time.perf_counter() - t) / 5
swap_s = busy = 0.0; misses = 0
tr = trace(); q = queue.Queue(); lats = []; start = time.perf_counter()
def feeder():
    for at, i in tr:
        d = start + at - time.perf_counter()
        if d > 0: time.sleep(d)
        q.put((at, i))
    q.put(None)
threading.Thread(target=feeder, daemon=True).start()
hit_lat, miss_lat = [], []
while (item := q.get()) is not None:
    at, i = item; was_hit = i in resident; serve(i)
    l = time.perf_counter() - start - at; lats.append(l); (hit_lat if was_hit else miss_lat).append(l)
wall = time.perf_counter() - start; lats.sort()
pct = lambda a, p: round(sorted(a)[min(len(a) - 1, int(len(a) * p))] * 1e3, 1) if a else None
print(json.dumps(dict(impl=impl, rate=rate, secs=secs, n=len(lats), service_ms=round(svc * 1e3, 1), misses=misses,
    miss_rate=round(misses / len(lats), 3), swap_ms_mean=round(swap_s / max(misses, 1) * 1e3, 1),
    swap_frac=round(swap_s / wall, 3), infer_frac=round(busy / wall, 3), p50_ms=pct(lats, .5), p99_ms=pct(lats, .99),
    max_ms=round(lats[-1] * 1e3, 1), miss_p50_ms=pct(miss_lat, .5), miss_p99_ms=pct(miss_lat, .99), hit_p99_ms=pct(hit_lat, .99))), flush=True)
