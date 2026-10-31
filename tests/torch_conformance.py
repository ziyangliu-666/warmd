# torch_conformance.py - PyTorch feature checks. Run natively and through warmd; each check prints
# one JSON line {"test", "ok", "value", ...}. tests/run.sh compares the two runs.
#   python3 torch_conformance.py [test ...]
import json, os, sys, threading, time
import torch

results = []
def check(name):
    def deco(fn):
        def run():
            t0 = time.perf_counter()
            try:
                v = fn(); ok = True; err = None
            except Exception as e:  # noqa
                v = None; ok = False; err = f"{type(e).__name__}: {e}"[:300]
            r = {"test": name, "ok": ok, "value": v, "ms": round((time.perf_counter() - t0) * 1e3, 1)}
            if err: r["error"] = err
            print(json.dumps(r), flush=True); results.append(r)
        run.__name__ = name; TESTS[name] = run; return run
    return deco
TESTS = {}
g = torch.Generator().manual_seed(1234)
A = torch.randn(512, 512, generator=g); B = torch.randn(512, 512, generator=g)
def r6(x): return round(float(x), 4)

@check("elementwise_reduce")
def _():
    x, y = A.cuda(), B.cuda(); return [r6((x + y).sum()), r6((x * y).mean()), r6(torch.relu(x).max())]

@check("cublas_matmul_fp32")
def _(): return r6((A.cuda() @ B.cuda()).abs().sum() / 1e4)

@check("cublas_matmul_fp16_tensorcore")
def _(): return r6((A.cuda().half() @ B.cuda().half()).float().abs().sum() / 1e4)

@check("cudnn_conv2d")
def _():
    torch.backends.cudnn.enabled = True
    x = torch.randn(4, 3, 64, 64, generator=g).cuda(); w = torch.randn(8, 3, 3, 3, generator=g).cuda()
    return r6(torch.nn.functional.conv2d(x, w, padding=1).abs().sum() / 1e3)

@check("softmax_layernorm_gelu")
def _():
    x = A.cuda(); y = torch.nn.functional.layer_norm(x, (512,)); return [r6(torch.softmax(x, 1)[:4].sum()), r6(torch.nn.functional.gelu(y).sum())]

@check("indexing_sort_topk_scatter")
def _():
    x = A.cuda()[:, 0]; s, i = torch.sort(x); t = torch.topk(x, 5).values
    z = torch.zeros(512, device="cuda").scatter_add_(0, i, s)
    return [r6(s[:3].sum()), r6(t.sum()), r6(z.sum())]

@check("large_copy_64MB_roundtrip")
def _():
    x = torch.arange(16 * 1024 * 1024, dtype=torch.float32)       # 64 MB > staging: chunked path
    y = x.cuda(); y.mul_(2); z = y.cpu(); return bool(torch.equal(z, x * 2))

@check("pinned_async_copies")
def _():
    src = torch.randn(1 << 20, generator=g).pin_memory(); dst = torch.empty_like(src).pin_memory()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        d = src.cuda(non_blocking=True); d.mul_(3); dst.copy_(d, non_blocking=True)
    s.synchronize()
    return bool(torch.allclose(dst, src * 3))

@check("streams_and_events")
def _():
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream(); e = torch.cuda.Event(enable_timing=True)
    x = A.cuda()
    with torch.cuda.stream(s1):
        y = x @ x; e.record()
    s2.wait_event(e)
    with torch.cuda.stream(s2):
        z = y + 1
    torch.cuda.synchronize(); return r6(z.abs().sum() / 1e4)

@check("cuda_graph_capture_replay")
def _():
    static_in = torch.zeros(512, 512, device="cuda"); w = B.cuda()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): static_out = torch.relu(static_in @ w)           # warm-up
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        static_out = torch.relu(static_in @ w)
    vals = []
    for k in range(3):
        static_in.copy_(A.cuda() * (k + 1)); gr.replay(); vals.append(r6(static_out.sum() / 1e4))
    return vals

@check("two_threads")
def _():
    out = [None, None]
    def work(i):
        torch.cuda.set_device(0); x = (A if i == 0 else B).cuda()
        for _ in range(20): x = torch.tanh(x @ x.T / 512)
        out[i] = r6(x.abs().sum())
    ts = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    [t.start() for t in ts]; [t.join() for t in ts]; return out

@check("memory_api")
def _():
    torch.cuda.empty_cache(); x = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    a = torch.cuda.memory_allocated(); del x; torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(); return [a >= (256 << 20), total > 0, free <= total]

@check("jiterator_nvrtc")
def _():
    fn = torch.cuda.jiterator._create_jit_fn("template <typename T> T my_op(T x, T y) { return x * y + T(1); }")
    return r6(fn(A.cuda(), B.cuda()).sum())

try:
    import triton, triton.language as tl
    @triton.jit
    def _tadd(x, y, o, n, BLOCK: tl.constexpr):
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = p < n
        tl.store(o + p, tl.load(x + p, mask=m) + tl.load(y + p, mask=m), mask=m)
except ImportError:
    triton = None

@check("triton_kernel")
def _():
    if triton is None: return "skipped: triton not installed"
    add = _tadd
    x, y = A.cuda().flatten(), B.cuda().flatten(); o = torch.empty_like(x)
    add[(triton.cdiv(x.numel(), 1024),)](x, y, o, x.numel(), BLOCK=1024)
    return r6(o.sum())

@check("device_properties")
def _():
    p = torch.cuda.get_device_properties(0); return [p.name, p.multi_processor_count, p.major, p.minor]

if __name__ == "__main__":
    names = sys.argv[1:] or list(TESTS)
    for n in names: TESTS[n]()
    sys.stdout.flush(); os._exit(0 if all(r["ok"] for r in results) else 1)
