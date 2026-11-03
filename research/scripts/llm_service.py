"""H26: which row of the capacity model fits LLM traffic? Uses the public Azure LLM inference traces.

A batched engine shares the GPU across requests, so the GPU seconds of a request are
context tokens / prefill rate + generated tokens / decode rate. The rates are assumptions, not
measurements. The mean compute per second of service time comes from results/h20_fleet.jsonl.
"""
import csv, json, statistics as st, sys
from datetime import datetime
RATES = [("low", 2000, 300), ("mid", 10000, 1500), ("high", 40000, 6000)]
UTIL, GB_USABLE = 0.7, 38
fleet = json.loads(open(sys.argv[1]).readline())["results"]
r = next(x for x in fleet if x["keep_min"] == 5 and x["svc_s"] == 10.0 and x["model_gb"] == 1.6)
gpus_per_svc_s = r["segments_mean"] / 10.0        # compute-bound there, so linear in service time
dedicated = r["dedicated_mean"]; resident = dedicated
out = dict(fleet_day=1, dedicated_mean_gpus=dedicated, compute_gpus_per_second_of_service=gpus_per_svc_s, traces=[])
for path in sys.argv[2:]:
    rows = list(csv.DictReader(open(path))); t = [datetime.fromisoformat(x["TIMESTAMP"][:26]) for x in rows]; c = [int(x["ContextTokens"]) for x in rows]; g = [int(x["GeneratedTokens"]) for x in rows]
    span = (max(t) - min(t)).total_seconds(); gaps = sorted((b - a).total_seconds() for a, b in zip(t, t[1:]))
    line = dict(trace=path.split("_")[-1][:-4], requests=len(rows), span_s=round(span), rate_per_s=round(len(rows) / span, 2), context_p50=st.median(c), generated_p50=st.median(g), context_mean=round(st.mean(c)), generated_mean=round(st.mean(g)), gap_p99_s=round(gaps[int(len(gaps) * .99)], 2), gap_max_s=round(gaps[-1], 1), by_rate=[])
    for name, p, d in RATES:
        svc = st.mean(c) / p + st.mean(g) / d; comp = gpus_per_svc_s * svc; e = dict(rates=name, prefill_tok_s=p, decode_tok_s=d, gpu_seconds_per_request=round(svc, 3))
        for gb in (1.6, 8.0, 16.0, 40.0):
            mem = resident * gb / GB_USABLE; need = max(comp, mem); e[f"saving_{gb}gb"] = round(dedicated * max(1, gb / GB_USABLE) / need, 1); e[f"limit_{gb}gb"] = "memory" if mem >= comp else "compute"
        line["by_rate"].append(e)
    out["traces"].append(line)
print(json.dumps(out, indent=1))
