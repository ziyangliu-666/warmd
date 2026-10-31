# Compatibility

Tested: A100-SXM4-40GB, driver 570.211.01, PyTorch 2.5.1+cu124, Triton 3.1. All 16 tests in
`tests/run.sh` pass; the 15 PyTorch checks and 11 driver API checks give the same results as
native CUDA. Other driver branches print a warning until they are tested.

## Works

- device queries and attributes (one device per daemon)
- primary and explicit contexts (virtual; one real context per daemon)
- allocation, free, memory info, pitched and stream-ordered allocation (allocated eagerly)
- copies in every direction, 2D copies, `cuMemcpy`, memset 8/16/32
- modules and libraries (`cuModuleLoad*`, `cuLibrary*`, `cuKernel*`), fatbin, cubin, PTX, globals
- `cuLink*` (link options are ignored)
- `cuLaunchKernel`, `cuLaunchKernelEx` with attributes; cooperative launch is implemented but untested
- streams, priorities, events, timing, waits, host callbacks
- stream capture and graph instantiate, launch, nodes and destroy
- pointer attributes and address ranges
- multiple threads (calls from one process are serialized)
- `fork()` after CUDA initialization
- the libraries PyTorch uses: cuBLAS, cuBLASLt, cuDNN, NVRTC, Triton

## Does not work

These return `CUDA_ERROR_NOT_SUPPORTED`. `WARMD_TRACE=1` lists the ones a program calls.

- managed memory and mapped pinned memory
- IPC, peer access, more than one GPU per process
- building graphs node by node, updating graph execs
- arrays, textures, surfaces
- graphics interop, external memory and semaphores
- virtual memory management and memory pools
- stream memory operations (`cuStreamWaitValue*` and similar)

## Differences

- Large copies are slower (see perf.md).
- `cuCtxSetLimit` can only raise a limit, since the context is shared.
- A device fault in one session ends all sessions of the daemon (they get error 46).
- `cuStreamGetFlags` returns 0.
