# pool_bench.py <zygotes> <ntasks>: ZygotePool throughput for short torch tasks (distinct inputs, verified).
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python")); sys.path.insert(0, os.path.join(HERE, ".."))
from warmd import ZygotePool
import pool_tasks
z, n = int(sys.argv[1]), int(sys.argv[2])
env = dict(os.environ); env["PYTHONPATH"] = os.path.join(HERE, "..") + os.pathsep + env.get("PYTHONPATH", "")
with ZygotePool(size=z, env=env, max_inflight=8) as pool:
    pool.map(pool_tasks.matmul_task, range(4 * z))                 # warm the code cache
    lat = []; t0 = time.perf_counter()
    futs = []
    for i in range(n):
        ts = time.perf_counter(); f = pool.submit(pool_tasks.matmul_task, i)
        f.add_done_callback(lambda _f, ts=ts: lat.append(time.perf_counter() - ts)); futs.append(f)
    outs = [f.result() for f in futs]; wall = time.perf_counter() - t0
lat.sort()
print(json.dumps({"label": "warmd", "mode": f"pool{z}", "n": n, "wall_s": round(wall, 2), "tasks_per_s": round(n / wall, 1),
                  "p50_ms": round(lat[len(lat) // 2] * 1e3, 1), "p99_ms": round(lat[int(len(lat) * .99)] * 1e3, 1),
                  "wrong": sum(not o["ok"] for o in outs)}))
