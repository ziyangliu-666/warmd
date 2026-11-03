"""H23: fault isolation across containers. A real PyTorch kernel writes into read-only weights.

The earlier fault test here was invalid: Triton had no C compiler, and the exception looked like a
fault. This version uses fill_, and checks that a store to an ordinary tensor works first.
  bystander <flag dir>   runs a forward pass, waits for the writer, and runs it again
  writer                 stores into its own memory, then into the segment
"""
import json, os, sys, time
import torch
sys.path.insert(0, "/workspace/exp")
import vf
from common import config, path
SOCK = "/x/vf.sock"; GB = 2; D = "/x/flags"
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
role = sys.argv[1]; vf.init(); sd = vf.Client(SOCK).open("m", path(GB)); key = next(k for k in sd if "q_proj.weight" in k)
if role == "writer":
    z = torch.zeros(16, device="cuda"); z.fill_(5.0); torch.cuda.synchronize(); control = float(z[0]) == 5.0; before = float(sd[key].flatten()[0]); s = time.perf_counter()
    try:
        sd[key].fill_(9.0); torch.cuda.synchronize(); r = "store_completed"
    except Exception as e:
        r = "faulted: " + str(e).split("\n")[0][:70]
    print(json.dumps(dict(role=role, ordinary_store_ok=control, result=r, report_ms=round((time.perf_counter() - s) * 1e3)))); open(D + "/done", "w").close()
else:
    m = build(sd); a = forward(m); open(D + "/ready", "w").close()
    while not os.path.exists(D + "/done"):
        time.sleep(0.2)
    b = forward(m); print(json.dumps(dict(role=role, same_output=a == b, value=b, weight=float(sd[key].flatten()[0]))))
