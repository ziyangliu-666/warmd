"""How many GPUs a set of isolated tenants needs, with private weights or shared segments.

    python3 placement_sim.py

Each tenant runs one isolated replica of one model. The memory costs are from Result 38
and Result 40 (Qwen2.5-1.5B, vLLM 0.8.5, A100 40 GB, gpu_memory_utilization 0.12); see the constants.
A shared replica also needs, once per (GPU, model), the segment, and once per GPU, the broker context.

Two placement rules:
- private first fit: put each tenant on the first GPU with room.
- shared, model affinity: put a tenant on a GPU that already holds its model's segment if one has room;
  otherwise on the GPU with the most free memory, which then loads the segment.
This counts memory only. Result 38 shows that compute limits active replicas well before memory does on a
busy trace, so the numbers apply to tenants that are mostly idle, as in Results 18 and 19.
"""
import json, os, random

GPU = 39.5
IDLE = os.environ.get("IDLE") == "1"
# Device memory per replica in GiB. Idle: after start-up (Result 38). Under load: the mean while seven
# replicas serve the Result 38 trace (Result 40). Result 39 first assumed a 1.4 GiB activation peak under
# load, taken from vLLM's start-up profile. Measured, the extra under load is 0.2 GiB.
PRIVATE = 3.91 if IDLE else 4.12
SHARED_REPLICA = 0.94 if IDLE else 1.12
SEGMENT, BROKER = 2.89, 0.42


def tenants(n, models, zipf, seed=0):
    r = random.Random(seed)
    w = [1 / (k + 1) ** zipf for k in range(models)]
    return [r.choices(range(models), weights=w)[0] for _ in range(n)]


def private(ts):
    free = []
    for _ in ts:
        for g in range(len(free)):
            if free[g] >= PRIVATE:
                free[g] -= PRIVATE
                break
        else:
            free.append(GPU - PRIVATE)
    return len(free)


def shared(ts):
    free, held = [], []                      # per GPU: free GiB, set of models with a segment there
    for m in sorted(ts, key=ts.count, reverse=True):     # place popular models first
        g = next((g for g in range(len(free)) if m in held[g] and free[g] >= SHARED_REPLICA), None)
        if g is None:
            need = SEGMENT + SHARED_REPLICA
            g = max((g for g in range(len(free)) if free[g] >= need), key=lambda g: free[g], default=None)
            if g is None:
                free.append(GPU - BROKER); held.append(set()); g = len(free) - 1
            free[g] -= SEGMENT; held[g].add(m)
        free[g] -= SHARED_REPLICA
    return len(free)


def main():
    rows = []
    for models in (1, 4, 16, 64, 256):
        for n in (64, 256, 1024):
            ts = tenants(n, models, zipf=1.0)
            p, s = private(ts), shared(ts)
            rows.append(dict(idle=IDLE, tenants=n, models=models, distinct_models_used=len(set(ts)),
                             gpus_private=p, gpus_shared=s, saving=round(p / s, 2)))
            print(json.dumps(rows[-1]))


if __name__ == "__main__":
    main()
