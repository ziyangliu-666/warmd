"""H6b: run PyTorch end to end on device-memory files.

Each test prints one JSON line:
  equal       a Llama forward pass over mapped weights equals one over ordinary weights; time per pass
  fault       a kernel store into a read-only mapping faults only the writer
  share       device memory for N processes that map one file
  park        unmap, broker evict, reload, remap at the same address: costs and correctness
"""
import json, multiprocessing as mp, os, subprocess, sys, time
import torch
import triton, triton.language as tl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vf
from common import config, path
SOCK = "/tmp/vf.sock"; GB = int(os.environ.get("GB", 8))

def emit(**kw): print(json.dumps(kw), flush=True)
def used():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])

def build(sd):
    from accelerate import init_empty_weights
    from transformers import LlamaForCausalLM
    with init_empty_weights():
        m = LlamaForCausalLM(config(GB))
    m.load_state_dict(sd, assign=True); m.eval(); m.model.rotary_emb.to("cuda"); return m

def forward(m, ids):
    with torch.inference_mode():
        out = m(ids).logits; torch.cuda.synchronize(); return float(out.float().sum())

def ids():
    return torch.arange(32, device="cuda").unsqueeze(0)

def timed(m, x, k=20):
    forward(m, x); t = time.perf_counter()
    for _ in range(k): forward(m, x)
    return (time.perf_counter() - t) / k * 1e3

def t_equal():
    from safetensors.torch import load_file
    vf.init(); c = vf.Client(SOCK); s = time.perf_counter(); sd = c.open("m", path(GB)); first = time.perf_counter() - s
    m = build(sd); x = ids(); a = forward(m, x); ms = timed(m, x)
    return dict(first_open_s=round(first, 3), mapped_sum=a, mapped_ms=round(ms, 2))

def t_equal_native():
    from safetensors.torch import load_file
    sd = {k: v.to("cuda") for k, v in load_file(path(GB)).items()}; m = build(sd); x = ids()
    return dict(native_sum=forward(m, x), native_ms=round(timed(m, x), 2))

def in_child(fn, *a):
    ctx = mp.get_context("spawn"); q = ctx.Queue(); p = ctx.Process(target=_wrap, args=(q, fn, a)); p.start(); r = q.get(); p.join(); return r
def _wrap(q, fn, a):
    try:
        q.put(fn(*a))
    except Exception as e:
        q.put(dict(exception=str(e)[:200]))

def writer(addr_key):
    @triton.jit
    def store(p, off, val):
        tl.store(p + off, val)
    vf.init(); c = vf.Client(SOCK); sd = c.open("m", path(GB)); t = sd[addr_key]; z = torch.zeros(16, device="cuda", dtype=torch.float32)
    before = float(t.flatten()[0])
    s = time.perf_counter()
    try:
        store[(1,)](z, (t.data_ptr() - z.data_ptr()) // 4, 9.0); torch.cuda.synchronize(); res = "store_completed"
    except Exception as e:
        res = "faulted: " + str(e)[:80]
    return dict(writer=res, report_ms=round((time.perf_counter() - s) * 1e3), before=before)

def t_fault():
    vf.init(); c = vf.Client(SOCK); sd = c.open("m", path(GB)); m = build(sd); x = ids(); a = forward(m, x)
    key = next(k for k in sd if "layers.0.self_attn.q_proj.weight" in k); head = float(sd[key].flatten()[0])
    r = in_child(writer, key); b = forward(m, x); head2 = float(sd[key].flatten()[0])
    return dict(**r, bystander_same_output=a == b, bystander_alive=True, weight_changed=head != head2)

def holder(q, go, keep):
    vf.init(); c = vf.Client(SOCK); sd = c.open("m", path(GB)); m = build(sd); x = ids(); q.put(forward(m, x)); go.wait(); keep.wait()

def t_share():
    base = used(); out = []; ctx = mp.get_context("spawn")
    for n in (1, 4, 8, 16):
        q, go, keep = ctx.Queue(), ctx.Event(), ctx.Event(); ps = [ctx.Process(target=holder, args=(q, go, keep)) for _ in range(n)]
        [p.start() for p in ps]; vals = [q.get() for _ in ps]; time.sleep(1); u = used(); go.set(); keep.set(); [p.join() for p in ps]
        out.append(dict(procs=n, used_mib=u, same_output=len(set(vals)) == 1)); time.sleep(1)
    return dict(base_mib=base, runs=out)

def t_park():
    vf.init(); c = vf.Client(SOCK); sd = c.open("m", path(GB)); m = build(sd); x = ids(); a = forward(m, x); u0 = used(); T = {}
    s = time.perf_counter(); c.park("m"); T["park_ms"] = (time.perf_counter() - s) * 1e3
    s = time.perf_counter(); c.resume("m"); T["resume_resident_ms"] = (time.perf_counter() - s) * 1e3
    T["same_after_resident_resume"] = forward(m, x) == a
    c.park("m"); s = time.perf_counter(); c.evict("m"); T["evict_ms"] = (time.perf_counter() - s) * 1e3; time.sleep(1); u1 = used()
    s = time.perf_counter(); c.resume("m", path(GB)); T["resume_reload_ms"] = (time.perf_counter() - s) * 1e3
    s = time.perf_counter(); b = forward(m, x); T["first_forward_after_reload_ms"] = (time.perf_counter() - s) * 1e3
    T["same_after_reload"] = a == b; T["used_mapped_mib"] = u0; T["used_evicted_mib"] = u1
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in T.items()}

if __name__ == "__main__":
    if sys.argv[1] == "broker":
        vf.Broker(SOCK).serve()
    else:
        fn = {"native": t_equal_native, "equal": t_equal, "fault": t_fault, "share": t_share, "park": t_park}[sys.argv[1]]
        print(json.dumps(dict(test=sys.argv[1], gb=GB, **fn())), flush=True)
