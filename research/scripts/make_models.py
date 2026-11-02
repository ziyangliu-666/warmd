"""Write one safetensors file per model size. Weight values do not affect load or kernel timing."""
import os, torch
from accelerate import init_empty_weights
from safetensors.torch import save_file
from transformers import LlamaForCausalLM
from common import LAYERS, config, path
for gb in LAYERS:
    if os.path.exists(path(gb)):
        continue
    os.makedirs(os.path.dirname(path(gb)), exist_ok=True)
    with init_empty_weights():
        m = LlamaForCausalLM(config(gb))
    sd = {}
    for k, p in m.state_dict().items():
        if p.dtype.is_floating_point:
            sd[k] = torch.empty(p.shape, dtype=torch.float16).normal_(0, 0.02)
        else:
            sd[k] = torch.zeros(p.shape, dtype=p.dtype)
    save_file(sd, path(gb))
    print(gb, sum(t.numel() * t.element_size() for t in sd.values()) / 1e9, "GB", flush=True)
