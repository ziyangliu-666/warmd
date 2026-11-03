"""Baseline for H4 and H7: scale-to-zero containers, simulated with the measured H1 cold start.

The simulation favours the baseline. Cold starts run in parallel with no CPU or disk contention,
and never hold the GPU, although a real cold start copies its weights. A request waits for its own
container, then for the GPU. Containers stay warm for `keep` idle seconds, and at most five are warm.
"""
import heapq, json, os, random, sys
rate, secs, cold, svc = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
N = 20
def trace():
    if os.environ.get("TRACE"):
        return [tuple(x) for x in json.load(open(os.environ["TRACE"]))]
    rng = random.Random(7); out = []
    w = [1 / (i + 1) for i in range(N)]; total = sum(w)
    for i, x in enumerate(w):
        r = rate * x / total; t = rng.expovariate(0.1)
        while t < secs:
            burst = rng.expovariate(1 / 10); on = r * 4; end = min(t + burst, secs)
            while t < end:
                out.append((t, i)); t += rng.expovariate(on)
            t = end + rng.expovariate(1 / 30)
    return sorted(out)
for keep in (5, 60, 300):
    ready, last = {}, {}; misses = 0; heap = []
    for t, i in trace():
        for j in [j for j in ready if t - last[j] > keep]:
            del ready[j]
        if i not in ready:
            misses += 1
            if len(ready) >= 5:
                del ready[min(ready, key=lambda j: last[j])]
            ready[i] = t + cold
        last[i] = t; heapq.heappush(heap, (max(t, ready[i]), t))
    gpu = 0.0; lats = []
    while heap:
        start, t = heapq.heappop(heap); gpu = max(gpu, start) + svc; lats.append(gpu - t)
    lats.sort()
    print(json.dumps(dict(impl="sim_scale_to_zero", keep_s=keep, n=len(lats), misses=misses, p50_ms=round(lats[len(lats) // 2] * 1e3, 1), p99_ms=round(lats[int(len(lats) * .99)] * 1e3, 1), max_ms=round(lats[-1] * 1e3, 1))))
