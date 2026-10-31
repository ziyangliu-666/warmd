"""Task functions used by tests/pool_tests.py and bench (must be importable by zygotes)."""
import torch


def matmul_task(seed, n=256):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=g); b = torch.randn(n, n, generator=g)
    x, y = a.cuda(), b.cuda()
    out = {"mm": float((x @ y).abs().sum()), "add": float((x + y).sum()), "softmax": float(torch.softmax(x, 1)[0, :8].sum())}
    ref = {"mm": float((a @ b).abs().sum()), "add": float((a + b).sum()), "softmax": float(torch.softmax(a, 1)[0, :8].sum())}
    ok = all(abs(out[k] - ref[k]) <= 1e-3 * max(1.0, abs(ref[k])) for k in ref)
    return {"seed": seed, "ok": ok}


def slow_task(seed, seconds=1.5):
    import time
    x = torch.randn(1024, 1024, device="cuda", generator=None); t = time.time(); n = 0
    while time.time() - t < seconds:
        x = torch.tanh(x @ x / 1024); torch.cuda.synchronize(); n += 1
    return matmul_task(seed)


def bad_task():
    """Writes through a wild pointer: a real sticky device fault (illegal memory access)."""
    import triton, triton.language as tl
    x = torch.zeros(16, device="cuda")
    _wild[(1,)](x)
    torch.cuda.synchronize()
    return "should not get here"


try:
    import triton, triton.language as tl

    @triton.jit
    def _wild(p):
        tl.store(p + (1 << 40), 1.0)
except ImportError:  # pragma: no cover
    _wild = None
