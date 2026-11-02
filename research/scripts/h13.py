"""H13: revocation. Can the broker reclaim device memory from a client that does not park?

  evict_mapped   the broker releases a file while a client maps it
  ckpt_mapped    cuda-checkpoint on a client that still maps an imported file
  kill           the client is killed; how fast does memory return?
"""
import json, multiprocessing as mp, os, signal, subprocess, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vf
from common import path
from h8 import build, forward, used
SOCK = "/tmp/vf.sock"; TOOL = "/workspace/exp/cuda-checkpoint"; GB = 8
def emit(**k): print(json.dumps(k), flush=True)

def child(conn):
    vf.init(); c = vf.Client(SOCK); m = build(c.open("r", path(GB)), GB); ref = forward(m); torch.cuda.synchronize(); conn.send(ref)
    while conn.recv() == "check":
        try:
            conn.send(forward(m) == ref)
        except Exception as e:
            conn.send(str(e)[:80])

def wait_mem(limit, secs=10):
    t = time.perf_counter()
    while time.perf_counter() - t < secs:
        if used() <= limit:
            return round((time.perf_counter() - t) * 1e3)
        time.sleep(0.05)
    return None

if __name__ == "__main__":
    ctx = mp.get_context("spawn"); br = vf.Client(SOCK); a, b = ctx.Pipe(); p = ctx.Process(target=child, args=(b,), daemon=True); p.start(); a.recv(); time.sleep(1); u0 = used()
    br.evict("r"); time.sleep(1); u1 = used(); a.send("check"); alive = a.recv()
    emit(test="evict_mapped", used_before=u0, used_after_broker_release=u1, client_output_still_correct=alive)
    t = time.perf_counter(); r = subprocess.run([TOOL, "--toggle", "--pid", str(p.pid)], capture_output=True, text=True, timeout=120); ck = time.perf_counter() - t; time.sleep(1); u2 = used(); st = subprocess.run([TOOL, "--get-state", "--pid", str(p.pid)], capture_output=True, text=True).stdout.strip()
    emit(test="ckpt_mapped", rc=r.returncode, err=(r.stderr + r.stdout)[:160], ms=round(ck * 1e3), used_after=u2, state=st)
    t = time.perf_counter(); r = subprocess.run([TOOL, "--toggle", "--pid", str(p.pid)], capture_output=True, text=True, timeout=120); u3 = used(); a.send("check"); same = a.poll(20) and a.recv()
    emit(test="restore_mapped", rc=r.returncode, ms=round((time.perf_counter() - t) * 1e3), used_after=u3, output_correct=same)
    os.kill(p.pid, signal.SIGKILL); p.join(); emit(test="kill", ms_until_memory_below_2gb=wait_mem(2000), used=used())
    os._exit(0)
