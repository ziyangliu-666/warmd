# Compatibility

Supported driver branch: 570. On other branches the client library prints a warning at
start-up (silence it with `WARMD_ALLOW_UNVALIDATED_DRIVER=1`).

## Works

- device queries and attributes (one device per daemon)
- primary and explicit contexts (virtual; one real context per daemon)
- allocation, free, memory info, pitched and stream-ordered allocation (allocated eagerly)
- copies in every direction, 2D copies, `cuMemcpy`, memset 8/16/32
- modules and libraries (`cuModuleLoad*`, `cuLibrary*`, `cuKernel*`), fatbin, cubin, PTX, globals
- `cuLink*` (link options are ignored)
- `cuLaunchKernel`, `cuLaunchKernelEx` with attributes
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
