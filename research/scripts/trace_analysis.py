"""H16: how much GPU residency would be idle under real long-tail arrivals?

Input: Azure Functions 2019 trace, per-minute invocations for 24 hours. These are CPU functions, so
the trace gives arrival shape only. Each function stands for one model. Service time is an assumption,
so several values are reported. A model stays resident until `keep` minutes pass with no request.
"""
import csv, json, random, sys
path, out = sys.argv[1], sys.argv[2]
KEEPS = (1, 5, 10, 60); SVC = (0.1, 1.0, 10.0)
rows = []
with open(path) as f:
    r = csv.reader(f); next(r)
    for row in r:
        rows.append([int(x) for x in row[4:]])
n = len(rows); totals = sorted((sum(c) for c in rows), reverse=True); grand = sum(totals)
active = [sum(1 for x in c if x) for c in rows]
res = dict(functions=n, invocations=grand, top_1pct_share=round(sum(totals[:n // 100]) / grand, 4), top_10pct_share=round(sum(totals[:n // 10]) / grand, 4),
           under_1_per_hour=round(sum(t < 24 for t in totals) / n, 3), under_1_per_minute=round(sum(t < 1440 for t in totals) / n, 3), median_daily=totals[n // 2])
def resident(c, keep):
    minutes = 0; colds = 0; idle = keep + 1
    for x in c:
        if x:
            colds += idle > keep; idle = 0
        else:
            idle += 1
        minutes += idle <= keep and (idle < keep + 1)
    return minutes, colds
groups = dict(all=lambda t: True, tail_under_1_per_minute=lambda t: t < 1440)
res["idle_share"] = []
for name, test in groups.items():
    sel = [c for c in rows if test(sum(c))]
    for keep in KEEPS:
        rm = cold = 0; per = [resident(c, keep) for c in sel]; rm = sum(p[0] for p in per); cold = sum(p[1] for p in per); inv = sum(sum(c) for c in sel)
        line = dict(group=name, functions=len(sel), keep_min=keep, resident_gpu_hours=round(rm / 60), cold_starts=cold, cold_per_invocation=round(cold / inv, 4))
        for s in SVC:
            busy = sum(min(sum(min(x * s / 60, 1) for x in c), p[0]) for c, p in zip(sel, per))
            line[f"idle_share_svc_{s}s"] = round(1 - busy / rm, 4)
        res["idle_share"].append(line)
# A real window for replay: twenty random functions active at that time, real timestamps within each minute.
rng = random.Random(11); start = 600; mins = 10
cand = [c for c in rows if 0 < sum(c[start:start + mins]) and max(c[start:start + mins]) <= 120]
chosen = rng.sample(cand, 20); trace = []
for i, c in enumerate(chosen):
    for m in range(mins):
        for _ in range(c[start + m]):
            trace.append((round(m * 60 + rng.random() * 60, 4), i))
trace.sort(); json.dump(trace, open(out, "w"))
res["replay"] = dict(requests=len(trace), secs=mins * 60, candidates=len(cand), per_model=sorted((sum(1 for t in trace if t[1] == i) for i in range(20)), reverse=True))
print(json.dumps(res, indent=1))
