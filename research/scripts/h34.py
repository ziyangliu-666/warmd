"""H34: many models on one A100, batched serving with vLLM, KV included.

    python3 h34.py seg|plain <trace.json> <seconds>

seg    16 vLLM processes stay alive. Weights live in segments (warmd.segments). At most ACTIVE
       processes hold weights and a KV pool. An idle one parks (unmaps weights, drops KV), and
       after HIB_AFTER seconds it is checkpointed with cuda-checkpoint. The broker is killed and
       restarted at KILL_AT seconds.
plain  One vLLM engine per model, with its own weights. At most ACTIVE engines exist; a request
       for another model kills the least recently used idle engine and starts a new one.

Both use the same KV pool size, the same trace and the same limit on active models.
"""
import gc, json, os, queue, subprocess, sys, threading, time
import multiprocessing as mp

N, ACTIVE = 16, int(os.environ.get("ACTIVE", "6"))
HIB_AFTER = float(os.environ.get("HIB_AFTER", "2"))
KILL_AT = float(os.environ.get("KILL_AT", "1e9"))     # kill and restart the broker at this second
BUDGET_GIB = os.environ.get("BUDGET_GIB", "18")   # six resident segments: about the memory of six plain engines
UTIL = float(os.environ.get("UTIL", "0.12"))
Q = "/workspace/qwen15"; SOCK = "/tmp/seg.sock"
F = os.environ.get("F", "/workspace/qwen15_vllm.safetensors")   # the segment file; /dev/shm avoids a slow disk
SWAP = float(os.environ.get("SWAP", "4"))                       # vLLM CPU swap space per process, GiB, pinned
KEEP_CPU = os.environ.get("KEEP_CPU", "1") == "1"   # park drops only the GPU KV pool; the pinned CPU swap space stays
TOOL = "/workspace/exp/cuda-checkpoint"; ROOT = "/workspace/warmd"


def worker(i, mode, conn):
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    import torch
    from vllm import LLM, SamplingParams
    t0 = time.perf_counter()
    llm = LLM(model=Q, gpu_memory_utilization=UTIL, enforce_eager=True, max_model_len=2048, seed=0, dtype="bfloat16", disable_log_stats=True, swap_space=SWAP)
    w = llm.llm_engine.model_executor.driver_worker
    c = None
    if mode == "seg":
        sys.path.insert(0, os.path.join(ROOT, "python"))
        from warmd.segments import Client
        m = w.model_runner.model
        names = dict(m.named_parameters(remove_duplicate=False)); first, alias = {}, {}
        for k, p in names.items():
            alias[k] = first.setdefault(id(p), k)
        if not os.path.exists(F + ".done"):     # setup starts models one at a time, so the first one writes it
            from safetensors.torch import save_file
            save_file({k: names[k].detach().cpu().clone() for k in first.values()}, F)
            open(F + ".done", "w").close()
        while not os.path.exists(F + ".done"):
            time.sleep(0.5)
        c = Client(SOCK)
        sd = c.open(f"m{i}")
        for k in names:
            mod, _, leaf = k.rpartition(".")
            m.get_submodule(mod)._parameters[leaf] = torch.nn.Parameter(sd[alias[k]], requires_grad=False)
        del names, first
        gc.collect(); torch.cuda.empty_cache()
    conn.send(("ready", time.perf_counter() - t0))
    while True:
        msg = conn.recv()
        if msg[0] == "gen":
            reqs = msg[1]
            prompts = [{"prompt_token_ids": ids} for _, ids, _ in reqs]
            params = [SamplingParams(max_tokens=n, temperature=0, ignore_eos=True) for _, _, n in reqs]
            outs = llm.generate(prompts, params, use_tqdm=False)
            conn.send(("done", time.time(), sum(len(o.outputs[0].token_ids) for o in outs)))
        elif msg[0] == "park":
            c.park(f"m{i}")
            if KEEP_CPU:
                for ce in w.cache_engine:
                    ce.gpu_cache = None
            else:
                w.cache_engine = None
            w.gpu_cache = None
            gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache(); conn.send(("parked",))
        elif msg[0] == "resume":
            t0 = time.perf_counter(); reloaded = c.resume(f"m{i}")
            t1 = time.perf_counter()
            if KEEP_CPU:
                for ce in w.cache_engine:
                    ce.gpu_cache = ce._allocate_kv_cache(ce.num_gpu_blocks, "cuda")
                w.gpu_cache = [ce.gpu_cache for ce in w.cache_engine]
            else:
                w._init_cache_engine()
            torch.cuda.synchronize()
            conn.send(("resumed", t1 - t0, time.perf_counter() - t1, bool(reloaded)))
        else:
            os._exit(0)


def used_mib():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])


def main():
    mode, trace_path, secs = sys.argv[1], sys.argv[2], float(sys.argv[3])
    trace = [tuple(x) for x in json.load(open(trace_path)) if x[0] < secs]
    ctx = mp.get_context("spawn")
    procs, pipes = [None] * N, [None] * N
    state = ["off"] * N            # off, active, parked, hib
    busy, last = [False] * N, [0.0] * N
    hib_threads = {}
    cond = threading.Condition()
    cold_lock = threading.Lock()   # plain: vLLM sizes its pool from free memory, so one engine changes at a time
    stats = dict(wakes=0, wake_s=[], colds=0, cold_s=[], restores=0, errors=0, tokens=0)
    parts = dict(wait=[], restore=[], map=[], kv=[], checkpoint=[])   # seconds, one entry per wake or checkpoint
    reloads = [0]
    broker = None

    def start_broker():
        env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "python"))
        if os.path.exists(SOCK):
            os.unlink(SOCK)
        files = sum((["--file", f"m{k}={F}"] for k in range(N)), [])
        p = subprocess.Popen([sys.executable, "-m", "warmd.segments", "--socket", SOCK, "--budget-gib", BUDGET_GIB] + files, env=env)
        while not os.path.exists(SOCK):
            time.sleep(0.05)
        return p

    def spawn(i):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(i, mode, b), daemon=True)
        p.start()
        if not a.poll(600):                 # a worker that died at import would block this forever
            raise RuntimeError(f"model {i} did not start")
        msg = a.recv()
        procs[i], pipes[i] = p, a
        return msg[1]

    def toggle(i, part=None):
        t = time.perf_counter()
        r = subprocess.run([TOOL, "--toggle", "--pid", str(procs[i].pid)], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"cuda-checkpoint on model {i}: {r.stdout} {r.stderr}")
        if part:
            parts[part].append(time.perf_counter() - t)

    def call(i, *msg):
        pipes[i].send(msg)
        return pipes[i].recv()

    if mode == "seg":
        broker = start_broker()
        # Start one at a time: vLLM sizes its pool from free memory. Start the idle models first,
        # so their segments are unmapped and the budget can evict them for the active ones.
        for i in reversed(range(N)):
            spawn(i)
            if i >= ACTIVE:
                call(i, "park"); toggle(i); state[i] = "hib"
            else:
                state[i] = "active"
    else:
        for i in range(ACTIVE):
            spawn(i); state[i] = "active"
    setup_used = used_mib()

    def make_active(i):
        """Called with cond held. Returns with model i active and busy."""
        held = False                       # plain: this thread holds cold_lock
        while True:
            if busy[i]:                    # another thread is parking or stopping this model
                cond.wait(); continue
            if state[i] == "active":
                busy[i] = True
                return
            n_active = sum(s in ("active", "waking") for s in state)   # a waking model already counts
            if n_active < ACTIVE:
                break
            idle = [k for k in range(N) if state[k] == "active" and not busy[k]]
            if not idle:
                cond.wait(); continue
            v = min(idle, key=lambda k: last[k]); busy[v] = True
            cond.release()
            try:
                if mode == "seg":
                    call(v, "park")
                else:
                    if not held:
                        cold_lock.acquire(); held = True
                    if procs[v].is_alive():
                        pipes[v].send(("stop",)); procs[v].join(60)
                    else:
                        print(f"engine {v} had exited with code {procs[v].exitcode}", file=sys.stderr, flush=True)
                        stats["dead"] = stats.get("dead", 0) + 1
                    if procs[v].is_alive():
                        procs[v].kill(); procs[v].join()
            finally:
                cond.acquire()
            state[v] = "parked" if mode == "seg" else "off"; busy[v] = False; last[v] = time.time()
        busy[i] = True; prev = state[i]; state[i] = "waking"
        cond.release()
        t = time.perf_counter()
        try:
            if mode == "seg":
                th = hib_threads.pop(i, None)
                tw = time.perf_counter()
                if th:
                    th.join()
                parts["wait"].append(time.perf_counter() - tw)
                if prev == "hib":
                    toggle(i, "restore"); stats["restores"] += 1
                r = call(i, "resume")
                parts["map"].append(r[1]); parts["kv"].append(r[2]); reloads[0] += r[3]
                stats["wakes"] += 1; stats["wake_s"].append(time.perf_counter() - t)
            else:
                try:
                    if not held:
                        cold_lock.acquire(); held = True
                    spawn(i)
                finally:
                    cold_lock.release()
                stats["colds"] += 1; stats["cold_s"].append(time.perf_counter() - t)
        finally:
            cond.acquire()
        state[i] = "active"

    def hibernator():
        while True:
            time.sleep(0.5)
            with cond:
                for k in range(N):
                    if state[k] == "parked" and not busy[k] and time.time() - last[k] > HIB_AFTER and k not in hib_threads:
                        th = threading.Thread(target=toggle, args=(k, "checkpoint")); th.start(); hib_threads[k] = th; state[k] = "hib"

    if mode == "seg":
        threading.Thread(target=hibernator, daemon=True).start()

    qs = [queue.Queue() for _ in range(N)]
    lat, done = [], threading.Event()
    rec = threading.Lock()
    start = time.time()

    def model_loop(i):
        while True:
            batch = [qs[i].get()]
            while True:
                try:
                    batch.append(qs[i].get_nowait())
                except queue.Empty:
                    break
            with cond:
                make_active(i)
            try:
                r = call(i, "gen", [(rid, ids, n) for rid, at, ids, n in batch])
                end, toks = r[1], r[2]
                stats["tokens"] += toks
            except Exception:
                stats["errors"] += len(batch); end = time.time()
            with cond:
                busy[i] = False; last[i] = time.time(); cond.notify_all()
            with rec:
                lat.extend(end - (start + at) for _, at, _, _ in batch)
                if len(lat) >= len(trace):
                    done.set()

    for i in range(N):
        threading.Thread(target=model_loop, args=(i,), daemon=True).start()
    samples = []

    def sampler():
        while not done.is_set():
            samples.append(used_mib()); time.sleep(2)
    threading.Thread(target=sampler, daemon=True).start()
    killed = False
    for rid, (at, i, ctx_len, gen_len) in enumerate(trace):
        d = start + at - time.time()
        if d > 0:
            time.sleep(d)
        if mode == "seg" and not killed and at >= KILL_AT:
            broker.kill(); broker.wait(); broker = start_broker(); killed = True
        qs[i].put((rid, at, [100 + (rid + j) % 30000 for j in range(ctx_len)], gen_len))
    done.wait(secs * 3)
    wall = time.time() - start
    lat.sort()
    pct = lambda p: round(lat[min(len(lat) - 1, int(len(lat) * p))], 2) if lat else None
    out = dict(mode=mode, models=N, active=ACTIVE, util=UTIL, requests=len(trace), completed=len(lat), errors=stats["errors"],
               tokens_per_s=round(stats["tokens"] / wall, 1), p50_s=pct(.5), p90_s=pct(.9), p99_s=pct(.99), max_s=pct(1.0),
               wakes=stats["wakes"], wake_mean_s=round(sum(stats["wake_s"]) / max(1, len(stats["wake_s"])), 3), restores=stats["restores"],
               colds=stats["colds"], cold_mean_s=round(sum(stats["cold_s"]) / max(1, len(stats["cold_s"])), 1),
               setup_used_mib=setup_used, used_mean_mib=round(sum(samples) / max(1, len(samples))), used_max_mib=max(samples or [0]), broker_killed=killed, dead_engines=stats.get("dead", 0),
               hib_after=HIB_AFTER, swap_gib=SWAP, keep_cpu=KEEP_CPU, budget_gib=BUDGET_GIB, reloads=reloads[0],
               parts={k: dict(n=len(v), mean=round(sum(v) / len(v), 3), p50=round(sorted(v)[len(v) // 2], 3), max=round(max(v), 3)) for k, v in parts.items() if v})
    print(json.dumps(out), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
