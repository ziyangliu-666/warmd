"""Warm-up functions for zygotes (run once per zygote, before any fork)."""


def torch_cuda():
    """Initialize CUDA, the caching allocator, cuBLAS/cuBLASLt and common kernels through warmd.
    Natively you must NOT use this (a forked child cannot use the parent's CUDA context)."""
    import torch
    x = torch.randn(256, 256, device="cuda")
    y = (x @ x).relu_().softmax(1)
    (x.half() @ x.half()).float().sum().item()
    y.sum().item()
    torch.cuda.synchronize()


def torch_import_only():
    """Import torch but keep CUDA uninitialized (safe for native CUDA as well)."""
    import torch  # noqa: F401
