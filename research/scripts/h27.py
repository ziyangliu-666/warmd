"""H27: the state ladder under runc and gVisor. The broker always runs in a runc container.

Predictions before the run: park, resume and evict pass in both. cuda-checkpoint passes in runc and
fails in gVisor, because nvproxy has no checkpoint code. Fork from a zygote passes in both.
  ladder  ckpt  fork  vllm
"""
import gc, json, os, subprocess, sys, time
t0 = time.perf_counter()
import torch
sys.path.insert(0, "/workspace/exp")
import vf
from common import config, path
SOCK = "/x/vf.sock"; TOOL = "/workspace/exp/cuda-checkpoint"; GB = 2; impl = sys.argv[2]
def used():
    f, t = torch.cuda.mem_get_info(); return (t - f) >> 20
def build(sd):
    from accelerate import init_empty_weights
    from transformers import LlamaForCausalLM
    with init_empty_weights():
        m = LlamaForCausalLM(config(GB))
    m.load_state_dict(sd, assign=True); m.eval(); m.model.rotary_emb.to("cuda"); return m
def forward(m):
    x = torch.arange(32, device="cuda").unsqueeze(0)
    with torch.inference_mode():
        return float(m(x).logits.float().sum())
def emit(**k): print(json.dumps(dict(impl=impl, **k)), flush=True)
def toggle():
    t = time.perf_counter(); r = subprocess.run([TOOL, "--toggle", "--pid", str(os.getpid())], capture_output=True, text=True, timeout=60); return r.returncode, (r.stdout + r.stderr).strip()[:150], round((time.perf_counter() - t) * 1e3)

def ladder():
    vf.init(); c = vf.Client(SOCK); t = time.perf_counter(); sd = c.open("m", path(GB)); o = time.perf_counter() - t; m = build(sd); a = forward(m); T = {}; T["used_mapped_mib"] = used(); torch.cuda.synchronize()
    x = time.perf_counter(); [forward(m) for _ in range(20)]; T["forward_ms"] = round((time.perf_counter() - x) / 20 * 1e3, 2)
    x = time.perf_counter(); c.park("m"); T["park_ms"] = round((time.perf_counter() - x) * 1e3, 2); x = time.perf_counter(); c.resume("m"); T["resume_ms"] = round((time.perf_counter() - x) * 1e3, 2); T["same_resident"] = forward(m) == a
    c.park("m"); c.evict("m"); time.sleep(1); T["used_evicted_mib"] = used(); x = time.perf_counter(); c.resume("m", path(GB)); T["reload_ms"] = round((time.perf_counter() - x) * 1e3, 1); x = time.perf_counter(); b = forward(m); T["first_forward_ms"] = round((time.perf_counter() - x) * 1e3, 2); T["same_reload"] = a == b
    emit(test="ladder", open_ms=round(o * 1e3, 1), **T)

def ckpt():
    vf.init(); c = vf.Client(SOCK); m = build(c.open("m", path(GB))); a = forward(m); torch.cuda.synchronize(); c.park("m"); gc.collect(); torch.cuda.empty_cache(); before = used()
    rc, out, ms = toggle()
    if rc:
        emit(test="ckpt", ok=False, rc=rc, out=out); return
    after = used() if False else None; rc2, out2, ms2 = toggle(); c.resume("m"); emit(test="ckpt", ok=rc2 == 0, used_before_mib=before, checkpoint_ms=ms, restore_ms=ms2, same=forward(m) == a, out=out2)

def fork():
    import accelerate, transformers
    from transformers import LlamaForCausalLM
    torch.set_num_threads(1)
    r0, w0 = os.pipe(); p = os.fork()
    if p == 0:
        vf.init(); os.write(w0, repr(forward(build(vf.Client(SOCK).open("m", path(GB)), ))).encode()); os._exit(0)
    ref = float(os.read(r0, 100)); os.waitpid(p, 0); out = []
    for _ in range(4):
        r, w = os.pipe(); s = time.perf_counter(); p = os.fork()
        if p == 0:
            vf.init(); c = time.perf_counter() - s; v = forward(build(vf.Client(SOCK).open("m", path(GB)))); torch.cuda.synchronize(); os.write(w, json.dumps(dict(context_s=round(c, 3), total_s=round(time.perf_counter() - s, 3), same=v == ref)).encode()); os._exit(0)
        d = os.read(r, 300); os.waitpid(p, 0); out.append(json.loads(d))
    emit(test="fork", import_s=round(time.perf_counter() - t0, 1), runs=out)

def vllm():
    from safetensors.torch import save_file
    from vllm import LLM, SamplingParams
    Q = "/x/qwen"; F = "/x/vllm_weights.safetensors"; P = ["The capital of France is", "Write a short poem about memory:", "def fibonacci(n):"]
    llm = LLM(model=Q, gpu_memory_utilization=0.5, enforce_eager=True, dtype="float16", max_model_len=512, seed=0); sp = SamplingParams(temperature=0, max_tokens=64)
    gen = lambda: [x.outputs[0].text for x in llm.generate(P, sp, use_tqdm=False)]
    a = gen(); cold = time.perf_counter() - t0; w = llm.llm_engine.model_executor.driver_worker; m = w.model_runner.model
    names = dict(m.named_parameters(remove_duplicate=False)); first = {}; alias = {}
    for k, p in names.items():
        alias[k] = first.setdefault(id(p), k)
    if not os.path.exists(F):
        save_file({k: names[k].detach().cpu().clone() for k in first.values()}, F)
    c = vf.Client(SOCK); sd = c.open("vl", F)
    for k in names:
        mod, _, leaf = k.rpartition("."); m.get_submodule(mod)._parameters[leaf] = torch.nn.Parameter(sd[alias[k]], requires_grad=False)
    del names, first; gc.collect(); torch.cuda.empty_cache(); t = time.perf_counter(); b = gen(); run = time.perf_counter() - t; used_run = used()
    s = time.perf_counter(); c.park("vl"); w.cache_engine = None; w.gpu_cache = None; gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache(); pk = time.perf_counter() - s; c.evict("vl"); time.sleep(1); idle = used()
    s = time.perf_counter(); c.resume("vl", F); w._init_cache_engine(); wake = time.perf_counter() - s; d = gen(); total = time.perf_counter() - s
    emit(test="vllm", cold_start_s=round(cold, 2), same_mapped=a == b, same_after_wake=a == d, sample=a[0][:40], generate_s=round(run, 3), used_running_mib=used_run, used_parked_evicted_mib=idle, park_ms=round(pk * 1e3), wake_ms=round(wake * 1e3), wake_to_result_s=round(total, 3))

{"ladder": ladder, "ckpt": ckpt, "fork": fork, "vllm": vllm}[sys.argv[1]]()
os._exit(0)
