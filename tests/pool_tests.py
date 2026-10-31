"""ZygotePool tests against a running warmd (env must have LD_LIBRARY_PATH + WARMD_SOCKET)."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from warmd import ZygotePool, TaskError
import pool_tasks

env = dict(os.environ); env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + env.get("PYTHONPATH", "")
results = []
def report(name, ok, **kw):
    r = {"test": name, "ok": bool(ok), **kw}; print(json.dumps(r), flush=True); results.append(r)

with ZygotePool(size=int(os.environ.get("ZYGOTES", "4")), env=env) as pool:
    # throughput + correctness with distinct inputs
    n = 256; t0 = time.perf_counter()
    outs = pool.map(pool_tasks.matmul_task, range(n))
    dt = time.perf_counter() - t0
    report("pool_throughput", all(o["ok"] for o in outs), tasks=n, tasks_per_s=round(n / dt, 1),
           zygote_warm_s=[round(z["warm_s"], 2) for z in pool.zygote_info()])

    # one culprit among bystanders: bystanders must succeed, culprit must fail with kind=fault
    if pool_tasks._wild is not None:
        futs = [pool.submit(pool_tasks.slow_task, i) for i in range(6)]
        time.sleep(0.5); bad = pool.submit(pool_tasks.bad_task)
        by = []
        for f in futs:
            try: by.append(f.result(120)["ok"])
            except TaskError as e: by.append(f"failed: {e.kind}: {e.message[:150]}")
        try: bad.result(120); culprit = "succeeded?!"
        except TaskError as e: culprit = e.kind
        after = pool.map(pool_tasks.matmul_task, range(32))
        report("pool_fault_quarantine", all(b is True for b in by) and culprit == "fault" and all(o["ok"] for o in after),
               bystanders=by, culprit=culprit, after_ok=all(o["ok"] for o in after), stats=dict(pool.stats))
sys.exit(0 if all(r["ok"] for r in results) else 1)
