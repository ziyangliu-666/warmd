"""Build the H34 trace: Azure LLM conversation lengths, with bursty Zipf arrivals over 16 models."""
import csv, json, random, sys
rows = list(csv.DictReader(open(sys.argv[1]))); out_path, secs, rate = sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
rng = random.Random(5); N = 16; w = [1 / (i + 1) for i in range(N)]; tot = sum(w); out = []
for i in range(N):
    r = rate * w[i] / tot; t = rng.expovariate(1 / 20)
    while t < secs:
        end = min(t + rng.expovariate(1 / 30), secs)          # bursts of about 30 s
        while t < end:
            x = rng.choice(rows); out.append((round(t, 3), i, min(int(x["ContextTokens"]), 1024), max(1, min(int(x["GeneratedTokens"]), 256)))); t += rng.expovariate(r * 3)
        t = end + rng.expovariate(1 / 60)                     # idle gaps of about 60 s
out.sort(); json.dump(out, open(out_path, "w")); print(len(out), "requests;", "per model:", [sum(1 for x in out if x[1] == i) for i in range(N)])
