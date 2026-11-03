"""H20: how many GPUs would real arrivals need? A capacity model, not a measurement.

Each function stands for one model, as in H16. The top 1% of functions by volume are left out, because
they suit dedicated serving. Per minute of the day:
  dedicated  one GPU for each resident model, with keep-alive `keep`
  segments   GPUs = max(compute / target utilization, resident weights / usable device memory)
Peak is what a fleet must own. Mean is what elastic capacity would pay for. Assumptions are arguments.
"""
import csv, json, sys
GPU_GB, UTIL, CTX_GB = 40, 0.7, 0.53
rows = []
with open(sys.argv[1]) as f:
    r = csv.reader(f); next(r)
    for row in r:
        rows.append([int(x) for x in row[4:]])
rows.sort(key=sum, reverse=True); tail = rows[len(rows) // 100:]; M = 1440
out = dict(day=sys.argv[1][-7:-4], functions=len(tail), invocations=sum(map(sum, tail)), results=[])
for keep in (1, 5, 10):
    res = [0] * M; load = [0] * M
    active = []
    for c in tail:
        idle = keep + 1; a = []
        for m, x in enumerate(c):
            idle = 0 if x else idle + 1
            if idle <= keep:
                res[m] += 1
            if x:
                a.append((m, x))
        active.append(a)
    for svc in (0.1, 1.0, 10.0):
        comp = [0.0] * M
        for a in active:
            for m, x in a:
                comp[m] += x * svc / 60
        for gb in (1.6, 8.0, 16.0):
            seg = [max(comp[m] / UTIL, res[m] * gb / (GPU_GB - 2)) for m in range(M)]
            ded = res; pk_d, mn_d, pk_s, mn_s = max(ded), sum(ded) / M, max(seg), sum(seg) / M
            mem_bound = sum(res[m] * gb / (GPU_GB - 2) >= comp[m] / UTIL for m in range(M)) / M
            out["results"].append(dict(keep_min=keep, svc_s=svc, model_gb=gb, dedicated_peak=pk_d, dedicated_mean=round(mn_d), segments_peak=round(pk_s), segments_mean=round(mn_s), peak_ratio=round(pk_d / pk_s, 1), mean_ratio=round(mn_d / mn_s, 1), memory_bound_share=round(mem_bound, 2)))
print(json.dumps(out))
