# sys_tests.py - end-to-end tests of warmd + the shim on a real GPU. Each test starts its own
# daemon (own socket) when it needs a special config. Prints one JSON line per test, exit 1 on failure.
#   python3 tests/sys_tests.py [test ...]
import ctypes, json, os, re, signal, socket, struct, subprocess, sys, tempfile, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
PY = sys.executable

class Daemon:
    port_ctr = 9450
    def __init__(self, name, *args):
        Daemon.port_ctr += 1
        self.port = Daemon.port_ctr
        self.sock = f"/tmp/warmd-test/{name}.sock"
        self.log = open(f"/tmp/warmd-test-{name}.log", "w+")
        self.p = subprocess.Popen([f"{BUILD}/warmd", "--socket", self.sock, "--metrics-port", str(self.port), *args],
                                  stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        self.wait_for("ACTIVE with")
    def logtext(self):
        self.log.flush(); self.log.seek(0); return self.log.read()
    def wait_for(self, pat, timeout=30, count=1):
        t = time.time()
        while time.time() - t < timeout:
            if self.logtext().count(pat) >= count: return True
            if self.p.poll() is not None: raise RuntimeError("daemon exited:\n" + self.logtext())
            time.sleep(0.05)
        raise TimeoutError(f"'{pat}' not seen in daemon log:\n" + self.logtext()[-2000:])
    def env(self, **kw):
        e = dict(os.environ, LD_LIBRARY_PATH=f"{BUILD}/lib", WARMD_SOCKET=self.sock); e.update(kw); return e
    def metrics(self):
        txt = urllib.request.urlopen(f"http://127.0.0.1:{self.port}/metrics", timeout=2).read().decode()
        m = {}
        for line in txt.splitlines():
            if line and not line.startswith("#"):
                k, v = line.rsplit(" ", 1); m[k] = float(v)
        return m
    def stop(self):
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
            try: self.p.wait(10)
            except subprocess.TimeoutExpired: self.p.kill()

def run(cmd, env, timeout=120, **kw):
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout, **kw)

TESTS, results = {}, []
def test(fn):
    TESTS[fn.__name__] = fn; return fn

def report(name, ok, **info):
    r = {"test": name, "ok": bool(ok), **info}; print(json.dumps(r), flush=True); results.append(r)

D = None   # shared default daemon
def default():
    global D
    if D is None: D = Daemon("default", "--sessions", "32", "--standbys", "1")
    return D

# ---------------------------------------------------------------------------------------------
@test
def driver_api_demo():
    d = default(); r = run([f"{BUILD}/demo"], d.env())
    out = json.loads(r.stdout.strip().splitlines()[-1])
    report("driver_api_demo", r.returncode == 0 and out["wrong"] == 0, steady_launch_sync_us=out["steady_launch_sync_us"], total_ms=out["total_ms"])

@test
def driver_features():
    d = default()
    nat = run([f"{BUILD}/driver_features_native"], dict(os.environ)); shm = run([f"{BUILD}/driver_features"], d.env())
    N = [json.loads(l) for l in nat.stdout.splitlines() if l.startswith("{")]
    S = [json.loads(l) for l in shm.stdout.splitlines() if l.startswith("{")]
    report("driver_features", N == S and all(x["ok"] for x in S), checks=len(S), passed=sum(x["ok"] for x in S),
           failed=[x["test"] for x in S if not x["ok"]])

@test
def config_file():
    cfgp = "/tmp/warmd-test/test.conf"
    open(cfgp, "w").write("# test config\nsessions = 3\nstandbys = 0\nquota-mb = 300\nlog-level = 2\n")
    d = Daemon("conf", "--config", cfgp)
    try:
        m = d.metrics(); r = run([PY, "-c", "import torch;print(torch.cuda.mem_get_info()[1]>>20)"], d.env())
        report("config_file", m["warmd_sessions_capacity"] == 3 and r.stdout.strip() == "300",
               capacity=m["warmd_sessions_capacity"], quota_seen_mb=r.stdout.strip())
    finally:
        d.stop()

@test
def torch_conformance():
    d = default()
    nat = run([PY, f"{ROOT}/tests/torch_conformance.py"], dict(os.environ), timeout=600)
    shm = run([PY, f"{ROOT}/tests/torch_conformance.py"], d.env(), timeout=600)
    N = {j["test"]: j for j in map(json.loads, filter(lambda l: l.startswith("{"), nat.stdout.splitlines()))}
    S = {j["test"]: j for j in map(json.loads, filter(lambda l: l.startswith("{"), shm.stdout.splitlines()))}
    bad = []
    for k, n in N.items():
        s = S.get(k)
        if not s or s["ok"] != n["ok"] or json.dumps(s["value"]) != json.dumps(n["value"]): bad.append(k)
    report("torch_conformance", not bad and len(S) == len(N), checks=len(N), passed_native=sum(j["ok"] for j in N.values()),
           identical=len(N) - len(bad), mismatched=bad)

INNOCENT = r'''
import torch, time, json, sys
x = torch.randn(1024, 1024, device="cuda"); t = time.time(); n = 0
try:
    while time.time() - t < float(sys.argv[1]):
        x = torch.tanh(x @ x / 1024); torch.cuda.synchronize(); n += 1
    print(json.dumps({"ok": True, "iters": n}))
except Exception as e:
    print(json.dumps({"ok": False, "iters": n, "err": str(e).splitlines()[0][:120]}))
'''

@test
def fault_failover():
    d = Daemon("fault", "--sessions", "16", "--standbys", "1")
    try:
        inn = [subprocess.Popen([PY, "-c", INNOCENT, "6"], env=d.env(), stdout=subprocess.PIPE, text=True) for _ in range(3)]
        time.sleep(3.0)                                         # innocents are mid-flight
        t0 = time.time(); bad = run([f"{BUILD}/demo"], d.env(DEMO_BAD="1")); t_bad = time.time() - t0
        badj = json.loads(bad.stdout.strip().splitlines()[-1])
        t1 = time.time(); recovered = None
        while time.time() - t1 < 10:                            # how fast does a NEW task succeed?
            r = run([f"{BUILD}/demo"], d.env())
            if r.returncode == 0: recovered = time.time() - t1; break
        outs = [json.loads(p.communicate(timeout=60)[0].strip().splitlines()[-1]) for p in inn]
        d.wait_for("ready in", count=3)                         # a replacement standby was started
        m = d.metrics()
        ok = (badj["sync"] != 0 and recovered is not None and m["warmd_failovers_total"] == 1
              and m['warmd_workers{state="active"}'] == 1)
        report("fault_failover", ok, culprit_sync_err=badj["sync"], culprit_ms=round(t_bad * 1e3),
               new_task_ok_after_ms=round(recovered * 1e3) if recovered else None,
               innocents=[{"ok": o["ok"], "err": o.get("err", "")[:60]} for o in outs],
               failovers=m["warmd_failovers_total"], standby_respawned=m['warmd_workers{state="standby"}'])
    finally:
        d.stop()

HOLD = r'''
import torch, sys, time
x = torch.empty(int(sys.argv[1]) << 20, dtype=torch.uint8, device="cuda"); x.fill_(1); torch.cuda.synchronize()
print("held", flush=True); time.sleep(60)
'''

@test
def reclaim_on_kill():
    d = default()
    before = d.metrics()["warmd_session_device_memory_bytes"]
    p = subprocess.Popen([PY, "-c", HOLD, "2048"], env=d.env(), stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "held"
    held = d.metrics()["warmd_session_device_memory_bytes"]
    p.send_signal(signal.SIGKILL); p.wait()
    t0 = time.time()
    while time.time() - t0 < 5:
        after = d.metrics()["warmd_session_device_memory_bytes"]
        if after <= before: break
        time.sleep(0.02)
    report("reclaim_on_kill", held - before >= 2 << 30 and after <= before, held_mb=round((held - before) / 2**20),
           after_mb=round((after - before) / 2**20), reclaim_ms=round((time.time() - t0) * 1e3))

QUOTA = r'''
import torch, json
free, total = torch.cuda.mem_get_info()
try:
    x = torch.empty(1 << 30, dtype=torch.uint8, device="cuda"); big = "allocated"
except torch.OutOfMemoryError: big = "oom"
y = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
print(json.dumps({"total_mb": total >> 20, "big": big, "small": "ok"}))
'''

@test
def memory_quota():
    d = Daemon("quota", "--sessions", "4", "--standbys", "0", "--quota-mb", "512")
    try:
        r = run([PY, "-c", QUOTA], d.env()); j = json.loads(r.stdout.strip().splitlines()[-1])
        report("memory_quota", j["total_mb"] == 512 and j["big"] == "oom" and j["small"] == "ok", **j)
    finally:
        d.stop()

def raw_hello(sockpath, version):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET); s.connect(sockpath)
    s.send(struct.pack("IIII", 0x31444d57, version, os.getpid(), 0))
    msg = s.recv(4096); s.close()
    status = struct.unpack_from("i", msg, 8)[0]
    reason = msg[-96:].split(b"\0")[0].decode()
    return status, reason

@test
def protocol_version_check():
    d = default(); st, why = raw_hello(d.sock, 999)
    report("protocol_version_check", st == 1, status=st, reason=why)

@test
def uid_allowlist():
    d = default()
    r, w = os.pipe(); pid = os.fork()
    if pid == 0:
        os.close(r); os.setgid(65534); os.setuid(65534)
        try: st, why = raw_hello(d.sock, 2)
        except Exception as e: st, why = -1, str(e)
        os.write(w, json.dumps([st, why]).encode()); os._exit(0)
    os.close(w); os.waitpid(pid, 0); st, why = json.loads(os.read(r, 4096))
    report("uid_allowlist", st == 2, status=st, reason=why)

@test
def session_capacity():
    d = Daemon("cap", "--sessions", "2", "--standbys", "0", "--session-wait-ms", "0")
    try:
        hold = [subprocess.Popen([PY, "-c", HOLD, "1"], env=d.env(), stdout=subprocess.PIPE, text=True) for _ in range(2)]
        for h in hold: h.stdout.readline()
        r = run([f"{BUILD}/demo"], d.env(WARMD_CONNECT_TIMEOUT_MS="200"))
        for h in hold: h.kill(); h.wait()
        time.sleep(0.3); r2 = run([f"{BUILD}/demo"], d.env())
        report("session_capacity", r.returncode != 0 and "all sessions in use" in r.stderr and r2.returncode == 0,
               third=r.stderr.strip().splitlines()[0][:100], after_release_ok=r2.returncode == 0)
    finally:
        d.stop()

@test
def session_queueing():
    d = Daemon("queue", "--sessions", "2", "--standbys", "0", "--session-wait-ms", "10000")
    try:
        hold = [subprocess.Popen([PY, "-c", HOLD, "1"], env=d.env(), stdout=subprocess.PIPE, text=True) for _ in range(2)]
        for h in hold: h.stdout.readline()
        t0 = time.time(); waiter = subprocess.Popen([f"{BUILD}/demo"], env=d.env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(1.0); queued = d.metrics()["warmd_sessions_waiting"]
        hold[0].kill(); hold[0].wait()
        out, err = waiter.communicate(timeout=30); waited = time.time() - t0
        hold[1].kill(); hold[1].wait()
        report("session_queueing", queued == 1 and waiter.returncode == 0, queued_while_full=queued, served_after_ms=round(waited * 1e3))
    finally:
        d.stop()

FREE_OTHER = r'''
import ctypes, sys
cu = ctypes.CDLL("libcuda.so.1"); cu.cuInit(0)
p = ctypes.c_ulonglong(0)
if sys.argv[1] == "alloc":
    cu.cuMemAlloc_v2(ctypes.byref(p), 1 << 20); print(p.value, flush=True); sys.stdin.readline()
else:
    print(cu.cuMemFree_v2(ctypes.c_ulonglong(int(sys.argv[2]))))
'''

@test
def isolation():
    d = default()
    a = subprocess.Popen([PY, "-c", FREE_OTHER, "alloc"], env=d.env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    ptr = a.stdout.readline().strip()
    r = run([PY, "-c", FREE_OTHER, "free", ptr], d.env())
    maps = open(f"/proc/{a.pid}/maps").read()
    nreg = len(set(l.split()[-1] for l in maps.splitlines() if "warmd-session" in l))
    seg = [l for l in maps.splitlines() if "warmd-session" in l]
    a.stdin.write("\n"); a.stdin.flush(); a.wait()
    report("isolation", r.stdout.strip() == "1" and len(seg) == 1, foreign_free_result=int(r.stdout.strip() or -1),
           session_mappings_in_client=len(seg))

@test
def host_restart_clean_error():
    d = Daemon("restart", "--sessions", "4", "--standbys", "0")
    p = subprocess.Popen([PY, "-c", INNOCENT, "20"], env=d.env(), stdout=subprocess.PIPE, text=True)
    time.sleep(3); d.p.send_signal(signal.SIGKILL); d.p.wait()
    # its workers notice the closed control socket and exit on their own
    t0 = time.time(); out = json.loads(p.communicate(timeout=30)[0].strip().splitlines()[-1])
    report("host_restart_clean_error", not out["ok"] and time.time() - t0 < 5, client_err=out.get("err", "")[:80],
           client_saw_error_after_ms=round((time.time() - t0) * 1e3))

@test
def warm_fork():
    d = default()
    r = run([PY, f"{ROOT}/bench/torch_e2e.py", "warm", "32", "4", "shim"], d.env(), timeout=300)
    j = json.loads(r.stdout.strip().splitlines()[-1])
    report("warm_fork", j["wrong"] == 0, tasks_per_s=j["tasks_per_s"], p50_ms=j["p50_ms"])

@test
def zygote_pool():
    d = Daemon("pool", "--sessions", "64", "--standbys", "1")
    try:
        r = run([PY, f"{ROOT}/tests/pool_tests.py"], d.env(), timeout=900)
        lines = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
        report("zygote_pool", r.returncode == 0 and len(lines) >= 1, results=lines, stderr=r.stderr[-300:] if r.returncode else "")
    finally:
        d.stop()

@test
def metrics_endpoint():
    d = default(); m = d.metrics()
    need = ["warmd_sessions_active", "warmd_rpc_total", "warmd_kernel_launches_total", "warmd_gpu_memory_total_bytes",
            "warmd_code_cache_hits_total", 'warmd_workers{state="active"}']
    miss = [k for k in need if k not in m]
    report("metrics_endpoint", not miss, missing=miss, rpc_total=m.get("warmd_rpc_total"), cache_hits=m.get("warmd_code_cache_hits_total"))

if __name__ == "__main__":
    os.makedirs("/tmp/warmd-test", exist_ok=True)
    names = sys.argv[1:] or list(TESTS)
    try:
        for n in names:
            try: TESTS[n]()
            except Exception as e: report(n, False, error=f"{type(e).__name__}: {e}"[:400])
    finally:
        if D: D.stop()
    sys.exit(0 if all(r["ok"] for r in results) else 1)
