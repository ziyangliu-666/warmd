"""H32: the cost of one hardware queue. Independent small kernels on N streams, and a model forward.

With CUDA_DEVICE_MAX_CONNECTIONS=1 all streams share one queue, so kernels from different streams may
serialize. Run once per setting, in a fresh process, because the driver reads it at init.
"""
import json, os, sys, time
import torch
label = os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "default")
def streams(n, size, secs=3.0):
    ss = [torch.cuda.Stream() for _ in range(n)]; xs = [torch.randn(size, size, device="cuda") for _ in range(n)]
    for s, x in zip(ss, xs):
        with torch.cuda.stream(s):
            x @ x
    torch.cuda.synchronize(); k = 0; end = time.perf_counter() + secs
    while time.perf_counter() < end:
        for s, x in zip(ss, xs):
            with torch.cuda.stream(s):
                for _ in range(8):
                    x @ x
        k += 8 * n
        if k % (64 * n) == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize(); return k / secs
out = dict(connections=label, device=torch.cuda.get_device_name())
for size in (128, 512):
    for n in (1, 4, 8):
        out[f"streams{n}_mm{size}_ops_s"] = round(streams(n, size))
sys.path.insert(0, "/workspace/exp")
from common import config
from accelerate import init_empty_weights
from transformers import LlamaForCausalLM
with init_empty_weights():
    m = LlamaForCausalLM(config(2))
m = m.to_empty(device="cuda").half().eval(); x = torch.arange(32, device="cuda").unsqueeze(0)
with torch.inference_mode():
    for _ in range(3):
        m(x)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20):
        m(x)
    torch.cuda.synchronize()
out["forward_ms"] = round((time.perf_counter() - t) / 20 * 1e3, 2); print(json.dumps(out), flush=True)
