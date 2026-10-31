"""ZygotePool: run many short GPU tasks, each in its own forked process, from pre-warmed zygotes.

With warmd + the shim, a zygote can import torch AND initialize CUDA/cuBLAS once; every task then
starts as fork() of that warm process (~10 ms) with no CUDA init. Natively this is impossible: a CUDA
context cannot cross fork().

Failure handling (one shared context => a device fault kills it for everyone):
  * a task failing with a "lost"/"fault" CUDA error is NOT failed immediately; it is quarantined
  * the pool drains, recycles the zygotes (their handles belong to the dead context), then re-runs
    quarantined tasks one at a time with nothing else on the GPU
  * a task that faults again alone is the culprit -> TaskError(kind="fault"); bystanders succeed
"""
import collections, itertools, json, os, selectors, socket, subprocess, sys, threading, time
from concurrent.futures import Future

from ._wire import recv_frame, send_frame

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TaskError(Exception):
    def __init__(self, kind, message, tb=""):
        super().__init__(f"[{kind}] {message}")
        self.kind, self.message, self.remote_traceback = kind, message, tb


class _Task:
    __slots__ = ("id", "fn", "args", "kwargs", "future", "quarantined", "lost_retries", "t_submit")
    def __init__(self, tid, fn, args, kwargs):
        self.id, self.fn, self.args, self.kwargs = tid, fn, args, kwargs
        self.future = Future(); self.quarantined = False; self.lost_retries = 0; self.t_submit = time.perf_counter()


class _Zygote:
    def __init__(self, pool, idx):
        self.idx = idx; self.inflight = set(); self.info = None
        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        env = dict(pool.env); env["PYTHONPATH"] = _PKG_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        cfg = {"preload": list(pool.preload), "warmup": pool.warmup, "torch_threads": pool.torch_threads}
        self.proc = subprocess.Popen([sys.executable, "-m", "warmd._zygote", str(b.fileno()), json.dumps(cfg)],
                                     pass_fds=(b.fileno(),), env=env)
        b.close(); self.sock = a

    def wait_ready(self, timeout):
        self.sock.settimeout(timeout)
        msg = recv_frame(self.sock)
        self.sock.settimeout(None)
        if not msg or msg[0] != "ready":
            raise RuntimeError(f"zygote {self.idx} failed to start (exit {self.proc.poll()})")
        self.info = msg[1]

    def kill(self):
        try: send_frame(self.sock, ("exit",))
        except OSError: pass
        try: self.proc.wait(2)
        except subprocess.TimeoutExpired: self.proc.kill(); self.proc.wait()
        self.sock.close()


class ZygotePool:
    def __init__(self, size=2, preload=("torch",), warmup="warmd.warm:torch_cuda", max_inflight=8,
                 env=None, retry=True, start_timeout=300, torch_threads=1):
        self.size, self.preload, self.warmup, self.max_inflight = size, tuple(preload), warmup, max_inflight
        self.env = dict(os.environ if env is None else env); self.retry, self.start_timeout = retry, start_timeout
        self.torch_threads = torch_threads
        self._ids = itertools.count(1); self._tasks = {}; self._queue = collections.deque(); self._quarantine = collections.deque()
        self._lock = threading.Lock(); self._recycle = False; self._closed = False
        self.stats = collections.Counter()
        self._zs = [_Zygote(self, i) for i in range(size)]
        for z in self._zs: z.wait_ready(start_timeout)
        self._wr, self._ww = os.pipe(); os.set_blocking(self._wr, False)
        self._thread = threading.Thread(target=self._loop, name="warmd-pool", daemon=True); self._thread.start()

    # ---------------- public API ----------------
    def submit(self, fn, *args, **kwargs):
        """fn: a module-level callable (pickled by reference) or a 'module:qualname' string."""
        if self._closed: raise RuntimeError("pool closed")
        t = _Task(next(self._ids), fn, args, kwargs)
        with self._lock:
            self._tasks[t.id] = t; self._queue.append(t)
        os.write(self._ww, b"x"); return t.future

    def map(self, fn, *iterables, timeout=None):
        futs = [self.submit(fn, *a) for a in zip(*iterables)]
        return [f.result(timeout) for f in futs]

    def zygote_info(self):
        return [z.info for z in self._zs]

    def close(self):
        if self._closed: return
        self._closed = True; os.write(self._ww, b"q"); self._thread.join(10)
        for z in self._zs: z.kill()

    def __enter__(self): return self
    def __exit__(self, *e): self.close()

    # ---------------- scheduler (single IO thread) ----------------
    def _inflight(self): return sum(len(z.inflight) for z in self._zs)

    def _send(self, z, t):
        z.inflight.add(t.id)
        try: send_frame(z.sock, ("run", t.id, t.fn, t.args, t.kwargs))
        except OSError: z.inflight.discard(t.id); self._queue.appendleft(t)

    def _dispatch(self):
        with self._lock:
            if self._recycle:
                if self._inflight(): return
                self.stats["recycles"] += 1
                for z in self._zs: z.kill()
                self._zs = [_Zygote(self, i) for i in range(self.size)]
                for z in self._zs: z.wait_ready(self.start_timeout)
                self._register_all(); self._recycle = False
            if self._quarantine:                     # run suspects alone, one at a time
                if self._inflight() == 0:
                    t = self._quarantine.popleft(); self.stats["quarantine_runs"] += 1; self._send(self._zs[0], t)
                return
            while self._queue:
                z = min(self._zs, key=lambda z: len(z.inflight))
                if len(z.inflight) >= self.max_inflight: break
                self._send(z, self._queue.popleft())

    def _register_all(self):
        for k in list(self._sel.get_map().values()):
            if k.data != "wake": self._sel.unregister(k.fileobj)
        for z in self._zs: self._sel.register(z.sock, selectors.EVENT_READ, z)

    def _on_done(self, z, msg):
        _, tid, status, payload, dt = msg
        z.inflight.discard(tid)
        t = self._tasks.get(tid)
        if t is None: return
        if status == "ok":
            self.stats["ok"] += 1; del self._tasks[tid]; t.future.set_result(payload); return
        kind, message, tb = payload
        if kind in ("lost", "fault", "crashed"):
            self._recycle = True                     # the shared context (and zygote handles) are gone
        if self.retry and kind in ("lost", "fault", "crashed") and (not t.quarantined or (kind == "lost" and t.lost_retries < 3)):
            # the shared context died (or this task faulted): drain, recycle zygotes, retry alone
            if t.quarantined: t.lost_retries += 1
            t.quarantined = True; self._quarantine.append(t); self._recycle = True; self.stats["suspects"] += 1
            return
        self.stats["failed"] += 1; del self._tasks[tid]
        t.future.set_exception(TaskError(kind, message, tb))

    def _loop(self):
        self._sel = selectors.DefaultSelector()
        self._sel.register(self._wr, selectors.EVENT_READ, "wake")
        self._register_all()
        while not self._closed:
            try: self._dispatch()
            except Exception as e:  # zygote restart failed: fail everything pending
                with self._lock:
                    for t in list(self._tasks.values()): t.future.set_exception(TaskError("pool", str(e)))
                    self._tasks.clear(); self._queue.clear(); self._quarantine.clear()
            for key, _ in self._sel.select(0.5):
                if key.data == "wake":
                    try:
                        if b"q" in os.read(self._wr, 4096): return
                    except BlockingIOError: pass
                    continue
                z = key.data
                msg = recv_frame(z.sock)
                if msg is None:                       # zygote died: its in-flight tasks are suspects
                    self._sel.unregister(z.sock)
                    with self._lock:
                        for tid in list(z.inflight):
                            self._on_done(z, ("done", tid, "err", ("crashed", "zygote died", ""), 0.0))
                        self._recycle = True
                    continue
                if msg[0] == "done":
                    with self._lock: self._on_done(z, msg)
