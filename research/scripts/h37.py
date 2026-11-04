"""H37: isolated replicas of one model on one GPU, with private weights or one shared segment.

    python3 h37.py cap-private|cap-shared                  start replicas until one fails; report how many fit
    python3 h37.py serve-one|serve-private|serve-shared <trace.json> <seconds>

cap-*    Each replica is its own vLLM process with gpu_memory_utilization UTIL. In cap-shared, every replica
         swaps its weights into the same segment, so the device holds one copy.
serve-*  R replicas (one for serve-one, with UTIL_ONE) serve the trace; request j goes to the j-th live
         replica in turn. At FAULT_AT seconds, replica 0 of serve-shared writes into its read-only
         weights. At the end every live replica answers a fixed greedy request, which is compared with
         its answer before the run.
"""
import gc, json, os, queue, subprocess, sys, threading, time
import multiprocessing as mp

UTIL = float(os.environ.get("UTIL", "0.12"))
UTIL_ONE = float(os.environ.get("UTIL_ONE", "0.85"))
R = int(os.environ.get("R", "6"))
SPEED = float(os.environ.get("SPEED", "1"))             # trace time is divided by this
FAULT_AT = float(os.environ.get("FAULT_AT", "1e9"))
SWAP = float(os.environ.get("SWAP", "2"))
Q = "/workspace/qwen15"; SOCK = "/tmp/seg.sock"; ROOT = "/workspace/warmd"
F = os.environ.get("F", "/dev/shm/qwen15_vllm.safetensors")
CHECK = [9707, 11, 1879, 0]


def worker(i, util, shared, conn):
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    import torch
    from vllm import LLM, SamplingParams
    try:
        import vllm.model_executor.models.registry as reg      # see h34.py: inspect in this process
        reg._run_in_subprocess = lambda fn: fn()
    except (ImportError, AttributeError):
        pass
    try:
        llm = LLM(model=Q, gpu_memory_utilization=util, enforce_eager=True, max_model_len=2048, seed=0,
                  dtype="bfloat16", disable_log_stats=True, swap_space=SWAP)
    except Exception as e:
        conn.send(("failed", repr(e)[:300])); return
    w = llm.llm_engine.model_executor.driver_worker
    w = getattr(w, "worker", w)
    m = w.model_runner.model
    if shared:
        sys.path.insert(0, os.path.join(ROOT, "python"))
        from warmd.segments import Client
        names = dict(m.named_parameters(remove_duplicate=False)); first, alias = {}, {}
        for k, p in names.items():
            alias[k] = first.setdefault(id(p), k)
        if not os.path.exists(F + ".done"):
            from safetensors.torch import save_file
            save_file({k: names[k].detach().cpu().clone() for k in first.values()}, F)
            open(F + ".done", "w").close()
        c = Client(SOCK)
        sd = c.open("m")                                    # every replica maps the same segment
        for k in names:
            mod, _, leaf = k.rpartition(".")
            m.get_submodule(mod)._parameters[leaf] = torch.nn.Parameter(sd[alias[k]], requires_grad=False)
        del names, first, p                                 # p would keep the tied embedding alive
        gc.collect(); torch.cuda.empty_cache()

    def answer():
        o = llm.generate([{"prompt_token_ids": CHECK}], SamplingParams(max_tokens=16, temperature=0), use_tqdm=False)
        return list(o[0].outputs[0].token_ids)
    conn.send(("ready", answer()))
    while True:
        msg = conn.recv()
        if msg[0] == "gen":
            prompts = [{"prompt_token_ids": ids} for _, ids, _ in msg[1]]
            params = [SamplingParams(max_tokens=n, temperature=0, ignore_eos=True) for _, _, n in msg[1]]
            outs = llm.generate(prompts, params, use_tqdm=False)
            conn.send(("done", time.time(), sum(len(o.outputs[0].token_ids) for o in outs)))
        elif msg[0] == "check":
            conn.send(("checked", answer()))
        elif msg[0] == "fault":                             # a kernel store into the read-only weights
            try:
                m.model.embed_tokens.weight.data.add_(1); torch.cuda.synchronize()
                conn.send(("fault", "no error"))
            except Exception as e:
                conn.send(("fault", repr(e)[:200]))
            os._exit(3)
        else:
            os._exit(0)


def used_mib():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])


def main():
    mode = sys.argv[1]
    ctx = mp.get_context("spawn")
    shared = mode.endswith("shared")
    broker = None
    if shared:
        env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "python"))
        if os.path.exists(SOCK):
            os.unlink(SOCK)
        broker = subprocess.Popen([sys.executable, "-m", "warmd.segments", "--socket", SOCK, "--file", f"m={F}"], env=env)
        while not os.path.exists(SOCK):
            time.sleep(0.05)
    procs, pipes, refs = [], [], []

    def spawn(util):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(len(procs), util, shared, b), daemon=True)
        p.start()
        deadline = time.time() + 600
        while time.time() < deadline and p.is_alive() and not a.poll(1):
            pass
        msg = a.recv() if a.poll(0) else ("failed", f"exit code {p.exitcode}")
        if msg[0] != "ready":
            p.kill(); p.join()
            return msg[1]
        procs.append(p); pipes.append(a); refs.append(msg[1])
        return None

    if mode.startswith("cap"):
        steps = []
        while True:
            err = spawn(UTIL)
            steps.append(dict(replicas=len(procs), used_mib=used_mib(), error=err))
            print(json.dumps(steps[-1]), flush=True)
            if err or len(procs) >= 40:
                break
        print(json.dumps(dict(mode=mode, util=UTIL, fit=len(procs), used_mib=steps[-1]["used_mib"],
                              same_output=all(r == refs[0] for r in refs), error=steps[-1]["error"])), flush=True)
        os._exit(0)

    trace_path, secs = sys.argv[2], float(sys.argv[3])
    trace = [(at / SPEED, c, g) for at, _, c, g in json.load(open(trace_path)) if at / SPEED < secs]
    n = 1 if mode == "serve-one" else R
    for _ in range(n):
        err = spawn(UTIL_ONE if mode == "serve-one" else UTIL)
        if err:
            print(json.dumps(dict(mode=mode, start_error=err, started=len(procs))), flush=True); os._exit(1)
    setup_used = used_mib()
    alive = [True] * n
    qs = [queue.Queue() for _ in range(n)]
    lat, errors, tokens = [], [0] * n, [0]
    rec, done = threading.Lock(), threading.Event()
    fault = {}
    start = time.time()

    def loop(k):
        while True:
            batch = [qs[k].get()]
            while True:
                try:
                    batch.append(qs[k].get_nowait())
                except queue.Empty:
                    break
            do_fault = batch[-1] == "FAULT"
            batch = [x for x in batch if x != "FAULT"]
            if batch:
                try:
                    pipes[k].send(("gen", [(rid, ids, g) for rid, at, ids, g in batch]))
                    r = pipes[k].recv(); end = r[1]; tokens[0] += r[2]
                except (EOFError, OSError, BrokenPipeError):
                    errors[k] += len(batch); end = time.time()
                with rec:
                    lat.extend(end - (start + at) for _, at, _, _ in batch)
                    if len(lat) >= len(trace):
                        done.set()
            if do_fault:
                # Only this thread uses this pipe, so the fault cannot mix with a request in flight.
                pipes[k].send(("fault",))
                try:
                    reply = pipes[k].recv()[1]
                except EOFError:
                    reply = "the process died before it could reply"
                procs[k].join(30)
                fault.update(reply=reply, exit_code=procs[k].exitcode)
                return

    for k in range(n):
        threading.Thread(target=loop, args=(k,), daemon=True).start()
    samples = []

    def sampler():
        while not done.is_set():
            samples.append(used_mib()); time.sleep(2)
    threading.Thread(target=sampler, daemon=True).start()
    nxt = 0
    for rid, (at, c, g) in enumerate(trace):
        d = start + at - time.time()
        if d > 0:
            time.sleep(d)
        if mode == "serve-shared" and alive[0] and at >= FAULT_AT:
            alive[0] = False; fault["at"] = round(at, 1)
            qs[0].put("FAULT")              # after the requests already queued for replica 0
        live = [k for k in range(n) if alive[k]]
        k = live[nxt % len(live)]; nxt += 1
        qs[k].put((rid, at, [100 + (rid + j) % 30000 for j in range(c)], g))
    done.wait(secs * 3)
    wall = time.time() - start
    same = {k: pipes[k].send(("check",)) or pipes[k].recv()[1] == refs[k] for k in range(n) if alive[k]}
    lat.sort()
    pct = lambda p: round(lat[min(len(lat) - 1, int(len(lat) * p))], 2) if lat else None
    print(json.dumps(dict(mode=mode, replicas=n, util=UTIL_ONE if n == 1 else UTIL, speed=SPEED, requests=len(trace),
                          completed=len(lat), errors_by_replica=errors, tokens_per_s=round(tokens[0] / wall, 1),
                          p50_s=pct(.5), p90_s=pct(.9), p99_s=pct(.99), setup_used_mib=setup_used,
                          used_mean_mib=round(sum(samples) / max(1, len(samples))), fault=fault,
                          same_output_after=same)), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
