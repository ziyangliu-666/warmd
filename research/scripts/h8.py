"""H8: properties a device-memory file system needs beyond the happy path.

  register   DMA rate from the page cache pinned in place, with no host copy
  fork       time from fork() to first result in a process that imported torch before any CUDA call
  scrub      does freshly created device memory expose what a previous process wrote?
  ckpt       does cuda-checkpoint release the context memory of a parked process, and resume work?
  zoo        real architectures on read-only weights: same output, no faults
"""
import ctypes, json, multiprocessing as mp, os, subprocess, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vf
from common import config, path
SOCK = "/tmp/vf.sock"
def used():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
def emit(**k): print(json.dumps(k), flush=True)

def build(sd, gb):
    from accelerate import init_empty_weights
    from transformers import LlamaForCausalLM
    with init_empty_weights():
        m = LlamaForCausalLM(config(gb))
    m.load_state_dict(sd, assign=True); m.eval(); m.model.rotary_emb.to("cuda"); return m

def forward(m):
    x = torch.arange(32, device="cuda").unsqueeze(0)
    with torch.inference_mode():
        return float(m(x).logits.float().sum())

def t_register():
    vf.init(); L = vf.LIB; L.vf_mmap_file.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
    for gb in (2, 8):
        n = os.path.getsize(path(gb)); p = ctypes.c_void_p(); t = time.perf_counter(); vf.ck(L.vf_mmap_file(path(gb).encode(), n, ctypes.byref(p)), "mmap"); populate = time.perf_counter() - t
        t = time.perf_counter(); r = vf.host_register(p.value, n); reg = time.perf_counter() - t
        size = vf.round_up(n); h = vf.U64(); vf.ck(L.vf_create(ctypes.c_size_t(size), ctypes.byref(h)), "create"); va = vf.reserve(size); vf.map_ro(va, size, h.value, ro=False)
        rates = []
        for _ in range(3):
            t = time.perf_counter(); vf.ck(L.vf_write(vf.U64(va), p, ctypes.c_size_t(n)), "write"); rates.append(round(n / 1e9 / (time.perf_counter() - t), 1))
        vf.unmap(va, size); vf.release(h.value); emit(test="register", gb=round(n / 1e9, 1), populate_ms=round(populate * 1e3, 1), register_result=r, register_ms=round(reg * 1e3, 1), gbs=rates)

def t_fork():
    from accelerate import init_empty_weights
    from transformers import LlamaForCausalLM
    for gb in (2, 8):
        for rep in range(4):
            r, w = os.pipe(); s = time.perf_counter(); pid = os.fork()
            if pid == 0:
                T = {}; a = time.perf_counter(); vf.init(); T["context"] = time.perf_counter() - a
                a = time.perf_counter(); c = vf.Client(SOCK); sd = c.open(f"fork{gb}", path(gb)); T["open"] = time.perf_counter() - a
                a = time.perf_counter(); m = build(sd, gb); T["build"] = time.perf_counter() - a
                a = time.perf_counter(); forward(m); torch.cuda.synchronize(); T["first_forward"] = time.perf_counter() - a; T["total"] = time.perf_counter() - s
                os.write(w, json.dumps({k: round(v, 3) for k, v in T.items()}).encode()); os._exit(0)
            os.close(w); data = os.read(r, 4096); os.close(r); os.waitpid(pid, 0); emit(test="fork", gb=gb, rep=rep, **json.loads(data))

def scrub_a(q):
    vf.init(); n = 64 << 20; h = vf.U64(); vf.ck(vf.LIB.vf_create(ctypes.c_size_t(n), ctypes.byref(h)), "create"); va = vf.reserve(n); vf.map_ro(va, n, h.value, ro=False)
    t = vf.tensor_at(va, [n // 4], 2, 32); t.fill_(7.0); torch.cuda.synchronize(); del t; vf.unmap(va, n); vf.release(h.value)
    h2 = vf.U64(); vf.ck(vf.LIB.vf_create(ctypes.c_size_t(n), ctypes.byref(h2)), "create"); va2 = vf.reserve(n); vf.map_ro(va2, n, h2.value, ro=False)
    t2 = vf.tensor_at(va2, [n // 4], 2, 32); q.put(int((t2 != 0).sum()))

def scrub_b(q):
    vf.init(); n = 40 << 30; n = 8 << 30; h = vf.U64(); vf.ck(vf.LIB.vf_create(ctypes.c_size_t(n), ctypes.byref(h)), "create"); va = vf.reserve(n); vf.map_ro(va, n, h.value, ro=False)
    t = vf.tensor_at(va, [n // 4], 2, 32); q.put(int((t != 0).sum()))

def t_scrub():
    ctx = mp.get_context("spawn"); q = ctx.Queue(); p = ctx.Process(target=scrub_a, args=(q,)); p.start(); same = q.get(); p.join()
    p = ctx.Process(target=scrub_b, args=(q,)); p.start(); other = q.get(); p.join()
    emit(test="scrub", nonzero_same_process_after_release=same, nonzero_new_process=other)

def ckpt_child(conn, gb):
    vf.init(); c = vf.Client(SOCK); name = f"ck{gb}"; sd = c.open(name, path(gb)); m = build(sd, gb); ref = forward(m); torch.cuda.synchronize(); c.park(name); conn.send(ref)
    conn.recv(); t = time.perf_counter(); c.resume(name, path(gb)); v = forward(m); torch.cuda.synchronize(); conn.send((v, time.perf_counter() - t)); conn.recv()

def t_ckpt():
    tool = os.environ["CKPT"]; ctx = mp.get_context("spawn"); a, b = ctx.Pipe(); base = used(); p = ctx.Process(target=ckpt_child, args=(b, 2)); p.start(); ref = a.recv(); time.sleep(1); u0 = used()
    t = time.perf_counter(); r1 = subprocess.run([tool, "--toggle", "--pid", str(p.pid)], capture_output=True, text=True); lock = time.perf_counter() - t; time.sleep(1); u1 = used()
    st = subprocess.run([tool, "--get-state", "--pid", str(p.pid)], capture_output=True, text=True).stdout.strip()
    t = time.perf_counter(); r2 = subprocess.run([tool, "--toggle", "--pid", str(p.pid)], capture_output=True, text=True); rest = time.perf_counter() - t; u2 = used()
    a.send(1); v, d = a.recv(); a.send(1); p.join()
    emit(test="ckpt", base=base, used_parked=u0, used_checkpointed=u1, used_restored=u2, state=st, checkpoint_ms=round(lock * 1e3), restore_ms=round(rest * 1e3), resume_and_forward_ms=round(d * 1e3, 1), same=v == ref, err=(r1.stderr + r2.stderr)[:200])

def zoo_cases():
    import transformers as T
    x = lambda: torch.arange(16, device="cuda").unsqueeze(0)
    return {
        "bert": (lambda: T.BertModel(T.BertConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=4, intermediate_size=256)), lambda m: m(x()).last_hidden_state, None),
        "gpt2": (lambda: T.GPT2LMHeadModel(T.GPT2Config(n_embd=128, n_layer=2, n_head=4)), lambda m: m(x()).logits, True),
        "llama": (lambda: T.LlamaForCausalLM(T.LlamaConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4, vocab_size=1000)), lambda m: m(x()).logits, True),
        "t5": (lambda: T.T5ForConditionalGeneration(T.T5Config(d_model=128, d_ff=256, num_layers=2, num_heads=4)), lambda m: m(input_ids=x(), decoder_input_ids=x()).logits, None),
        "whisper": (lambda: T.WhisperForConditionalGeneration(T.WhisperConfig(d_model=64, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64)), lambda m: m(input_features=torch.ones(1, 80, 3000, device="cuda"), decoder_input_ids=x()).logits, None),
        "vit": (lambda: T.ViTModel(T.ViTConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=4, intermediate_size=256, image_size=64)), lambda m: m(torch.ones(1, 3, 64, 64, device="cuda")).last_hidden_state, None),
        "resnet_batchnorm": (lambda: T.ResNetModel(T.ResNetConfig(depths=[1, 1], hidden_sizes=[32, 64], layer_type="basic")), lambda m: m(torch.ones(1, 3, 64, 64, device="cuda")).last_hidden_state, None),
    }

def zoo_child(name, q):
    try:
        from safetensors.torch import save_file
        make, run, gen = zoo_cases()[name]; torch.manual_seed(0); m = make().eval(); vf.init()
        names = dict(m.named_parameters(remove_duplicate=False)); first = {}; alias = {}
        for k, p in names.items():
            alias[k] = first.setdefault(id(p), k)
        f = f"/tmp/zoo_{name}.safetensors"; save_file({k: names[k].detach().clone() for k in first.values()}, f)
        ref = m.to("cuda")
        with torch.inference_mode():
            a = run(ref).float().cpu(); control = float((a - run(ref).float().cpu()).abs().max()); g1 = ref.generate(torch.arange(4, device="cuda").unsqueeze(0), max_new_tokens=6, do_sample=False).cpu() if gen else None
        c = vf.Client(SOCK); sd = c.open("zoo_" + name, f)
        for k in names:
            mod, _, leaf = k.rpartition("."); o = m.get_submodule(mod); o._parameters[leaf] = torch.nn.Parameter(sd[alias[k]], requires_grad=False)
        with torch.inference_mode():
            b = run(m).float().cpu(); g2 = m.generate(torch.arange(4, device="cuda").unsqueeze(0), max_new_tokens=6, do_sample=False).cpu() if gen else None
        torch.cuda.synchronize(); q.put(dict(model=name, params=len(first), max_abs_diff=float((a - b).abs().max()), control_same_weights_diff=control, generate_same=None if not gen else bool((g1 == g2).all())))
    except Exception as e:
        q.put(dict(model=name, exception=str(e)[:220]))

def t_zoo():
    ctx = mp.get_context("spawn")
    for name in zoo_cases():
        q = ctx.Queue(); p = ctx.Process(target=zoo_child, args=(name, q)); p.start(); emit(test="zoo", **q.get()); p.join()

if __name__ == "__main__":
    {"register": t_register, "fork": t_fork, "scrub": t_scrub, "ckpt": t_ckpt, "zoo": t_zoo}[sys.argv[1]]()
