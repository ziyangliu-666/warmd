"""H15: a real checkpoint and a real serving engine on read-only segments.

  hf     Qwen2.5-0.5B, transformers generate: ordinary weights against mapped weights
  vllm   the same model in vLLM: replace its weights with a segment, then generate, park and resume
  fd     what an exported fd is, and whether a process can import it after the exporter exits
"""
import ctypes, json, multiprocessing as mp, os, socket, subprocess, sys, time
import torch
sys.path.insert(0, "/workspace/exp")
import vf
SOCK = "/tmp/vf.sock"; Q = "/workspace/qwen"; FILE = Q + "/model.safetensors"
PROMPTS = ["The capital of France is", "Write a short poem about memory:", "def fibonacci(n):"]
def used():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
def emit(**k): print(json.dumps(k), flush=True)

def t_hf():
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(Q)
    def gen(m):
        out = []; t = time.perf_counter()
        for p in PROMPTS:
            x = tok(p, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                out.append(tok.decode(m.generate(**x, max_new_tokens=40, do_sample=False)[0]))
        return out, time.perf_counter() - t
    from transformers import GenerationConfig
    probe = tok(PROMPTS[1], return_tensors="pt").to("cuda")
    def logits(m):
        with torch.inference_mode():
            return m(**probe).logits.float().cpu()
    m = AutoModelForCausalLM.from_pretrained(Q, torch_dtype=torch.bfloat16).to("cuda").eval(); a, ta = gen(m); a, ta = gen(m); la = logits(m); del m; torch.cuda.empty_cache()
    c = vf.Client(SOCK); vf.init(); sd = c.open("qwen", FILE); cfg = AutoConfig.from_pretrained(Q)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg)
    m.load_state_dict(sd, assign=True, strict=False); m.tie_weights(); m.eval(); m.model.rotary_emb.to("cuda")
    no_cfg, _ = gen(m); lb = logits(m); m.generation_config = GenerationConfig.from_pretrained(Q)   # from_config does not read this file.
    b, _ = gen(m); b, tb = gen(m)
    emit(test="real_checkpoint_hf", dtype=str(next(iter(sd.values())).dtype), logits_max_abs_diff=float((la - lb).abs().max()), same_text_without_generation_config=a == no_cfg, same_text=a == b, sample=b[0][:70], native_s=round(ta, 3), mapped_s=round(tb, 3))

def t_vllm():
    from safetensors.torch import save_file
    from vllm import LLM, SamplingParams
    base = used(); llm = LLM(model=Q, gpu_memory_utilization=0.15, enforce_eager=True, dtype="float16", max_model_len=512, seed=0)
    sp = SamplingParams(temperature=0, max_tokens=64)
    def gen():
        t = time.perf_counter(); o = [x.outputs[0].text for x in llm.generate(PROMPTS, sp, use_tqdm=False)]; return o, time.perf_counter() - t
    a, _ = gen(); a, ta = gen(); u0 = used()
    m = llm.llm_engine.model_executor.driver_worker.model_runner.model
    names = dict(m.named_parameters(remove_duplicate=False)); first = {}; alias = {}
    for k, p in names.items():
        alias[k] = first.setdefault(id(p), k)
    f = "/tmp/vllm_weights.safetensors"; save_file({k: names[k].detach().cpu().clone() for k in first.values()}, f)
    size = sum(names[k].numel() * names[k].element_size() for k in first.values())
    c = vf.Client(SOCK); sd = c.open("vl", f)
    for k in names:
        mod, _, leaf = k.rpartition("."); m.get_submodule(mod)._parameters[leaf] = torch.nn.Parameter(sd[alias[k]], requires_grad=False)
    del names, first; import gc; gc.collect(); torch.cuda.empty_cache(); time.sleep(1); u1 = used()
    b, _ = gen(); b, tb = gen()
    c.park("vl"); c.evict("vl"); time.sleep(1); u2 = used(); s = time.perf_counter(); c.resume("vl", f); res = time.perf_counter() - s
    d, _ = gen()
    emit(test="vllm", same_text=a == b, same_after_evict_resume=a == d, sample=b[0][:70], ordinary_s=round(ta, 3), mapped_s=round(tb, 3), weights_mib=size >> 20, used_base=base, used_ordinary=u0, used_mapped_with_broker=u1, used_parked_evicted=u2, resume_reload_ms=round(res * 1e3, 1))

def fd_owner(path):
    vf.init(); n = 64 << 20; h = vf.U64(); vf.ck(vf.LIB.vf_create(ctypes.c_size_t(n), ctypes.byref(h)), "create"); va = vf.reserve(n); vf.map_ro(va, n, h.value, ro=False)
    t = vf.tensor_at(va, [n // 4], 2, 32); t.fill_(3.0); torch.cuda.synchronize(); fd = ctypes.c_int(); vf.ck(vf.LIB.vf_export(vf.U64(h.value), ctypes.byref(fd)), "export")
    s = socket.socket(socket.AF_UNIX); s.connect(path); socket.send_fds(s, [b"x"], [fd.value]); s.recv(1)
    del t; vf.unmap(va, n); vf.release(h.value); os._exit(0)     # The exporter is gone before the import.

def t_fd():
    ctx = mp.get_context("spawn"); path = "/tmp/fdtest.sock"
    if os.path.exists(path):
        os.unlink(path)
    l = socket.socket(socket.AF_UNIX); l.bind(path); l.listen(1); p = ctx.Process(target=fd_owner, args=(path,)); p.start(); c, _ = l.accept(); _, fds, _, _ = socket.recv_fds(c, 1, 1); fd = fds[0]
    st = os.fstat(fd); link = os.readlink(f"/proc/self/fd/{fd}"); kind = f"major={os.major(st.st_rdev)} minor={os.minor(st.st_rdev)}"; c.send(b"g"); p.join(); time.sleep(1)
    vf.init(); out = dict(test="fd", link=link, rdev=kind, mode=oct(st.st_mode))
    try:
        h = vf.import_fd(fd); va = vf.reserve(64 << 20); vf.map_ro(va, 64 << 20, h); t = vf.tensor_at(va, [(64 << 20) // 4], 2, 32); out.update(import_after_exporter_exit="ok", all_three=bool((t == 3.0).all()))
    except Exception as e:
        out.update(import_after_exporter_exit=str(e)[:120])
    emit(**out)

if __name__ == "__main__":
    {"hf": t_hf, "vllm": t_vllm, "fd": t_fd}[sys.argv[1]]()
