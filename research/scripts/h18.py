"""H18: the full state ladder on a real engine. vLLM keeps a KV pool, so release it at park.

Measures, for one process: engine start, then idle device memory and time to the same output after
  parked       weights unmapped, KV pool released
  hibernated   as parked, then cuda-checkpoint
  evicted      as hibernated, and the broker frees the segment
"""
import gc, json, os, subprocess, sys, time
t0 = time.perf_counter()
import torch
sys.path.insert(0, "/workspace/exp")
import vf
from safetensors.torch import save_file
from vllm import LLM, SamplingParams
Q = "/workspace/qwen"; SOCK = "/tmp/vf.sock"; TOOL = "/workspace/exp/cuda-checkpoint"; F = "/tmp/vllm_weights.safetensors"
PROMPTS = ["The capital of France is", "Write a short poem about memory:", "def fibonacci(n):"]
def used():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
def toggle():
    subprocess.run([TOOL, "--toggle", "--pid", str(os.getpid())], check=True, capture_output=True)

llm = LLM(model=Q, gpu_memory_utilization=0.15, enforce_eager=True, dtype="float16", max_model_len=512, seed=0)
sp = SamplingParams(temperature=0, max_tokens=64)
def gen():
    return [x.outputs[0].text for x in llm.generate(PROMPTS, sp, use_tqdm=False)]
a = gen(); cold_s = time.perf_counter() - t0
w = llm.llm_engine.model_executor.driver_worker; m = w.model_runner.model
names = dict(m.named_parameters(remove_duplicate=False)); first = {}; alias = {}
for k, p in names.items():
    alias[k] = first.setdefault(id(p), k)
save_file({k: names[k].detach().cpu().clone() for k in first.values()}, F)
c = vf.Client(SOCK); sd = c.open("vl", F)
for k in names:
    mod, _, leaf = k.rpartition("."); m.get_submodule(mod)._parameters[leaf] = torch.nn.Parameter(sd[alias[k]], requires_grad=False)
del names, first; gc.collect(); torch.cuda.empty_cache()

def park():
    s = time.perf_counter(); c.park("vl"); w.cache_engine = None; w.gpu_cache = None; gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache(); return time.perf_counter() - s
def wake():
    s = time.perf_counter(); c.resume("vl", F); r = time.perf_counter() - s; s = time.perf_counter(); w._init_cache_engine(); return r, time.perf_counter() - s

out = dict(test="ladder_vllm", engine_cold_start_s=round(cold_s, 2), segment_same=gen() == a, used_running_mib=used())
res = {}
for state in ("parked", "hibernated", "evicted"):
    p = park(); ck = 0.0
    if state != "parked":
        s = time.perf_counter(); toggle(); ck = time.perf_counter() - s
    if state == "evicted":
        c.evict("vl")
    time.sleep(1); idle = used(); s = time.perf_counter()
    if state != "parked":
        toggle()
    restore = time.perf_counter() - s; seg, kv = wake(); s2 = time.perf_counter(); b = gen(); g = time.perf_counter() - s2
    res[state] = dict(park_ms=round(p * 1e3, 1), checkpoint_ms=round(ck * 1e3), idle_used_mib=idle, restore_ms=round(restore * 1e3), segment_ms=round(seg * 1e3, 1), kv_ms=round(kv * 1e3, 1), generate_ms=round(g * 1e3), wake_to_result_ms=round((time.perf_counter() - s) * 1e3), same=b == a)
s = time.perf_counter(); b = gen(); out["running_generate_ms"] = round((time.perf_counter() - s) * 1e3); out.update(res)
print(json.dumps(out), flush=True); os._exit(0)
