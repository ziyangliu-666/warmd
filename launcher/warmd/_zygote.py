"""Zygote process: pre-imports (and optionally pre-warms CUDA through warmd), then forks one child
per task. Talks to its ZygotePool over an inherited socket with length-prefixed pickle frames.

Started as:  python -m warmd._zygote <fd> <json-config>
"""
import importlib, json, os, pickle, selectors, signal, socket, struct, sys, time, traceback

from ._wire import recv_frame, send_frame, resolve, classify_error


def _classify(e):
    try:   # ask the warmd shim (exact); fall back to the error text (e.g. native CUDA)
        import ctypes
        st = ctypes.CDLL("libcuda.so.1").cuWarmdStatus()
        if st == 46: return "lost"
        if st: return "fault"
    except (OSError, AttributeError):
        pass
    return classify_error(e)


def _child(fn, args, kwargs, w, torch_mod):
    try:
        if torch_mod is not None and getattr(torch_mod.cuda, "_initialized", False):
            # torch refuses CUDA in forked children because a native context cannot cross fork().
            # With warmd the process holds no CUDA state, so the child may continue.
            torch_mod.cuda._is_in_bad_fork = lambda: False
        t0 = time.perf_counter()
        res = fn(*args, **kwargs)
        out = ("ok", res, time.perf_counter() - t0)
    except BaseException as e:  # noqa
        out = ("err", (_classify(e), f"{type(e).__name__}: {e}"[:2000], traceback.format_exc()[-4000:]), 0.0)
    try:
        data = pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:  # unpicklable result
        data = pickle.dumps(("err", ("error", f"result not picklable: {e}", ""), 0.0))
    view = memoryview(data)
    while view:
        n = os.write(w, view); view = view[n:]
    os._exit(0)


def main():
    fd = int(sys.argv[1]); cfg = json.loads(sys.argv[2])
    sock = socket.socket(fileno=fd)
    t0 = time.perf_counter()
    torch_mod = None
    for m in cfg.get("preload", []):
        mod = importlib.import_module(m)
        if m == "torch": torch_mod = mod
    if torch_mod is not None:
        torch_mod.set_num_threads(cfg.get("torch_threads", 1))   # an OpenMP pool used before fork() deadlocks children
    if cfg.get("warmup"):
        resolve(cfg["warmup"])()
    if torch_mod is not None and torch_mod.cuda.is_initialized():
        torch_mod.cuda.synchronize()
        torch_mod._C._cuda_clearCublasWorkspaces()   # children must not share the parent's cuBLAS workspace
        torch_mod.cuda.empty_cache()                 # ...or its cached allocator blocks
    send_frame(sock, ("ready", {"pid": os.getpid(), "warm_s": time.perf_counter() - t0,
                                "cuda_warm": bool(torch_mod is not None and torch_mod.cuda.is_initialized())}))
    sel = selectors.DefaultSelector()
    sel.register(sock, selectors.EVENT_READ, ("ctl", None))
    running = {}   # pipe fd -> (task_id, pid, chunks)
    while True:
        for key, _ in sel.select():
            kind, info = key.data
            if kind == "ctl":
                msg = recv_frame(sock)
                if msg is None or msg[0] == "exit":
                    for (_, pid, _) in running.values():
                        try: os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError: pass
                    os._exit(0)
                _, task_id, fn_ref, args, kwargs = msg
                r, w = os.pipe()
                pid = os.fork()
                if pid == 0:
                    os.close(r); sel.close(); sock.close()
                    try: fn = resolve(fn_ref) if isinstance(fn_ref, str) else fn_ref
                    except BaseException as e:  # noqa
                        fn = None; err = e
                    if fn is None:
                        os.write(w, pickle.dumps(("err", ("error", f"cannot resolve {fn_ref}: {err}", ""), 0.0))); os._exit(0)
                    _child(fn, args, kwargs, w, torch_mod)
                os.close(w)
                os.set_blocking(r, False)
                running[r] = (task_id, pid, [])
                sel.register(r, selectors.EVENT_READ, ("task", r))
            else:
                r = info; task_id, pid, chunks = running[r]
                try:
                    b = os.read(r, 1 << 20)
                except BlockingIOError:
                    continue
                if b:
                    chunks.append(b); continue
                sel.unregister(r); os.close(r); del running[r]
                _, status = os.waitpid(pid, 0)
                if chunks:
                    out = pickle.loads(b"".join(chunks))
                else:
                    sig = os.WTERMSIG(status) if os.WIFSIGNALED(status) else None
                    out = ("err", ("crashed", f"task process died (signal {sig}, status {status})", ""), 0.0)
                send_frame(sock, ("done", task_id) + tuple(out))


if __name__ == "__main__":
    main()
