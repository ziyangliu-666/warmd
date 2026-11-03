"""H1: time each cold-start stage in a fresh process. Run natively and through warmd."""
import os, sys, time
t = time.perf_counter
gb, impl, cache = int(sys.argv[1]), sys.argv[2], sys.argv[3]
T = {}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import config, emit, path
if cache == "cold":
    fd = os.open(path(gb), os.O_RDONLY); os.fsync(fd); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
t0 = t()
import torch
T["import_torch"] = t() - t0
s = t(); from transformers import LlamaForCausalLM; from accelerate import init_empty_weights; from safetensors.torch import load_file
T["import_libs"] = t() - s
s = t(); torch.zeros(1, device="cuda"); torch.cuda.synchronize(); T["cuda_init_ctx"] = t() - s
s = t(); sd = load_file(path(gb), device="cpu"); T["read_weights"] = t() - s
for v in sd.values():
    pass
s = t()
sd = {k: v.to("cuda") for k, v in sd.items()}
torch.cuda.synchronize(); T["weights_to_gpu"] = t() - s
s = t()
with init_empty_weights():
    m = LlamaForCausalLM(config(gb))
m.load_state_dict(sd, assign=True); m.eval()
m.model.rotary_emb.to("cuda")
T["build_model"] = t() - s
ids = torch.randint(0, 32000, (1, 32), device="cuda")
with torch.inference_mode():
    s = t(); m(ids); torch.cuda.synchronize(); T["first_inference"] = t() - s
    s = t(); m(ids); torch.cuda.synchronize(); T["second_inference"] = t() - s
T["total_to_first_result"] = t() - t0
emit(impl=impl, gb=gb, cache=cache, **{k: round(v, 3) for k, v in T.items()})
