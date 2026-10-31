# torch_task.py - a small but real PyTorch GPU task. Prints one JSON line with a checksum so
# native and shim runs can be compared for correctness.
#   python3 torch_task.py            (native)
#   LD_LIBRARY_PATH=build/lib WARMD_SOCKET=/tmp/warmd/gpu0.sock python3 torch_task.py   (through warmd)
import json, os, sys, time
t0 = time.perf_counter()
import torch
t1 = time.perf_counter()
level = int(os.environ.get("TT_LEVEL", "9"))
g = torch.Generator().manual_seed(0)
a = torch.randn(256, 256, generator=g)
b = torch.randn(256, 256, generator=g)
out = {}
x = a.cuda(); y = b.cuda()                       # H2D (pageable)
t2 = time.perf_counter()
if level >= 1: z = x + y; out["add"] = float(z.sum().item())          # elementwise + reduction + D2H
if level >= 2: z = torch.relu(x) * 2.0; out["relu"] = float(z.mean().item())
if level >= 3: z = x @ y; out["mm"] = float(z.abs().sum().item())        # cuBLAS
if level >= 4: z = torch.softmax(x, dim=1); out["softmax"] = float(z[0, :8].sum().item())
if level >= 5: out["cpu"] = float((x * y).cpu().double().sum())           # D2H whole tensor
torch.cuda.synchronize()
t3 = time.perf_counter()
out.update(import_ms=round((t1 - t0) * 1e3, 1), init_h2d_ms=round((t2 - t1) * 1e3, 1),
           compute_ms=round((t3 - t2) * 1e3, 1), total_ms=round((t3 - t0) * 1e3, 1))
print(json.dumps(out)); sys.stdout.flush()
if not os.environ.get("TT_CLEAN_EXIT"):
    os._exit(0)  # skip slow python/CUDA teardown in both modes alike
