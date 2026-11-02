"""H9: the rate limit of hibernation. warmd exists because the driver creates about 5 contexts per second.

N parked processes are checkpointed. They are then restored one at a time, and all at once.
Reports restores per second, device memory, host memory per process, and output correctness.
"""
import json, multiprocessing as mp, os, subprocess, sys, threading, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vf
from common import config, path
from h8 import build, forward, used
SOCK = "/tmp/vf.sock"; TOOL = "/workspace/exp/cuda-checkpoint"; N = int(sys.argv[1]); GB = 2

def child(conn):
    vf.init(); c = vf.Client(SOCK); sd = c.open("h9", path(GB)); m = build(sd, GB); ref = forward(m); torch.cuda.synchronize(); c.park("h9"); conn.send(ref)
    while conn.recv() == "go":
        t = time.perf_counter(); c.resume("h9"); v = forward(m); torch.cuda.synchronize(); c.park("h9"); conn.send((v, time.perf_counter() - t))

def toggle(p):
    subprocess.run([TOOL, "--toggle", "--pid", str(p)], check=True, capture_output=True)

def rss_mib(p):
    for l in open(f"/proc/{p}/status"):
        if l.startswith("VmRSS"):
            return int(l.split()[1]) / 1024

def together(pids):
    ts = [threading.Thread(target=toggle, args=(p,)) for p in pids]; t = time.perf_counter(); [x.start() for x in ts]; [x.join() for x in ts]; return time.perf_counter() - t

if __name__ == "__main__":
    ctx = mp.get_context("spawn"); pipes, procs, refs = [], [], []; base = used()
    for _ in range(N):
        a, b = ctx.Pipe(); p = ctx.Process(target=child, args=(b,), daemon=True); p.start(); pipes.append(a); procs.append(p); refs.append(a.recv())
    pids = [p.pid for p in procs]; time.sleep(1); parked = used(); rss0 = sum(map(rss_mib, pids)) / N
    t_ck_par = together(pids); time.sleep(1); hib = used(); rss1 = sum(map(rss_mib, pids)) / N
    t = time.perf_counter(); [toggle(p) for p in pids]; seq_restore = time.perf_counter() - t
    t = time.perf_counter(); [toggle(p) for p in pids]; seq_ckpt = time.perf_counter() - t
    par_restore = together(pids); after = used()
    wrong = 0; ds = []
    for a, r in zip(pipes, refs):
        a.send("go"); v, d = a.recv(); wrong += v != r; ds.append(d)
    print(json.dumps(dict(test="hibernate_rate", procs=N, base_mib=base, parked_mib=parked, hibernated_mib=hib, restored_mib=after, freed_per_proc_mib=round((parked - hib) / N, 1), rss_parked_mib=round(rss0), rss_hibernated_mib=round(rss1),
        checkpoint_all_parallel_s=round(t_ck_par, 3), restore_sequential_per_s=round(N / seq_restore, 2), checkpoint_sequential_per_s=round(N / seq_ckpt, 2), restore_parallel_per_s=round(N / par_restore, 2), wrong=wrong, serve_ms_mean=round(sum(ds) / N * 1e3, 1))), flush=True)
    os._exit(0)
