"""H12: is the segment orthogonal to the context? One process per task, on a 7.5 GB model.

  warmd_copy   warmd today: each forked task copies the weights to the device
  warmd_seg    warmd with a persistent segment: the zygote prepared the model, and tasks only run it
  native_vf    fork, new native context, read-only device-memory file; tasks stay isolated
Each task is a new process that runs one forward pass, and its output is checked.
"""
import ctypes, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import config, path
mode, N, conc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); GB = 8; SOCK = "/tmp/vf.sock"
torch.set_num_threads(1)
from accelerate import init_empty_weights
from safetensors.torch import load_file
from transformers import LlamaForCausalLM

def build(sd):
    with init_empty_weights():
        m = LlamaForCausalLM(config(GB))
    m.load_state_dict(sd, assign=True); m.eval(); m.model.rotary_emb.to("cuda"); return m
def forward(m):
    x = torch.arange(32, device="cuda").unsqueeze(0)
    with torch.inference_mode():
        return float(m(x).logits.float().sum())
def settle():
    torch.cuda.synchronize(); torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache()

if mode == "warmd_copy":
    cpu = load_file(path(GB)); forward_ref = forward(build({k: v.to("cuda") for k, v in cpu.items()})); settle()
    def task():
        return forward(build({k: v.to("cuda") for k, v in cpu.items()}))
elif mode == "warmd_seg":
    import vf_tensor
    lib = ctypes.CDLL("libcuda.so.1"); cpu = load_file(path(GB)); layout = {}; pos = 0
    for k, v in cpu.items():
        pos = (pos + 4095) & ~4095; layout[k] = pos; pos += v.numel() * v.element_size()
    va, created = ctypes.c_uint64(), ctypes.c_int(); r = lib.cuWarmdSegOpen(ctypes.c_uint64(7), ctypes.c_size_t(pos), ctypes.byref(va), ctypes.byref(created)); assert r == 0, r
    sd = {}
    for k, v in cpu.items():
        t = vf_tensor.tensor_at(va.value + layout[k], list(v.shape), 0 if not v.is_floating_point() else 2, v.element_size() * 8)
        if created.value:
            t.copy_(v.to("cuda"))
        sd[k] = t
    # The zygote must hold no allocator memory. A live block in the PyTorch allocator is inherited by
    # every fork, so children place scratch at the same device address. In one shared context that
    # raises no error: at concurrency 2 or more, all outputs were NaN.
    del cpu; m = build(sd); forward_ref = forward(m); m.model.rotary_emb.to("cpu"); settle()
    def task():
        m.model.rotary_emb.to("cuda"); return forward(m)
else:
    import vf
    # The broker loads the file on first open. A throwaway child does this, and the parent stays free of CUDA.
    r, w = os.pipe(); pid = os.fork()
    if pid == 0:
        vf.init(); sd = vf.Client(SOCK).open("h12", path(GB)); os.write(w, repr(forward(build(sd))).encode()); os._exit(0)
    forward_ref = float(os.read(r, 100)); os.waitpid(pid, 0)
    def task():
        vf.init(); return forward(build(vf.Client(SOCK).open("h12", path(GB))))

vals = []; running = {}; lat = []; wrong = 0; i = 0; t0 = time.perf_counter()
while i < N or running:
    while i < N and len(running) < conc:
        r, w = os.pipe(); ts = time.perf_counter(); pid = os.fork()
        if pid == 0:
            os.close(r); torch.cuda._is_in_bad_fork = lambda: False
            try:
                os.write(w, repr(task()).encode())
            except Exception as e:
                os.write(w, ("ERR " + str(e)[:100]).encode())
            os._exit(0)
        os.close(w); running[pid] = (ts, r); i += 1
    pid, _ = os.wait(); ts, r = running.pop(pid); d = os.read(r, 200); os.close(r); lat.append(time.perf_counter() - ts)
    try:
        vals.append(float(d)); wrong += float(d) != forward_ref
    except ValueError:
        wrong += 1; print(d[:120], file=sys.stderr)
wall = time.perf_counter() - t0; lat.sort()
print(json.dumps(dict(test="task_per_process", mode=mode, gb=7.5, n=N, conc=conc, tasks_s=round(N / wall, 2), p50_ms=round(lat[len(lat) // 2] * 1e3, 1), p99_ms=round(lat[min(len(lat) - 1, int(len(lat) * .99))] * 1e3, 1), wrong=wrong, ref=forward_ref, distinct=sorted(set(vals))[:4], n_distinct=len(set(vals)))), flush=True)
