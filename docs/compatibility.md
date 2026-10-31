# Compatibility

## Tested environments

| GPU | driver | PyTorch and libraries | result |
|---|---|---|---|
| A100-SXM4-40GB | 570.211.01 (CUDA 12.8) | PyTorch 2.5.1+cu124, cuBLAS 12.3, cuDNN 9, Triton 3.1 | 16/16 system tests; 15 PyTorch and 11 driver API checks match native results |

Requirements: Linux on x86-64 and an NVIDIA driver 550 or newer. The worker uses
`cuFuncGetParamInfo`, which needs a CUDA 12.4 driver. The shim provides undocumented driver
tables that can change between driver branches, so it prints a warning on branches other than
570 until someone validates them ([new-driver.md](new-driver.md)).

## Supported driver API

| area | status | notes |
|---|---|---|
| initialization, device queries and attributes | yes | cached per process; each daemon exposes one device |
| primary and explicit contexts | yes | one real context in the daemon; client handles are virtual |
| allocation, free, memory info, pitched allocation | yes | free is checked against the session's own allocations |
| stream-ordered allocation | yes | allocates immediately; frees after the stream drains |
| copies host to device, device to host, device to device, 2D, `cuMemcpy` | yes | async copies with pinned memory are asynchronous up to a quarter of `ring-mb` |
| memset 8, 16 and 32 bit | yes | |
| modules and libraries (`cuModuleLoad*`, `cuLibrary*`, `cuKernel*`), globals | yes | fatbin, cubin and PTX; one shared cache per daemon |
| JIT linking (`cuLink*`) | yes | link options are ignored |
| `cuLaunchKernel`, `cuLaunchKernelEx` with attributes | yes | |
| cooperative launch | implemented | not tested |
| streams, priorities, events, timing, queries, waits | yes | default-stream semantics are kept within a session |
| stream capture and CUDA graphs: begin, end, instantiate, launch, nodes, destroy | yes | capture runs in relaxed mode on the daemon |
| host callbacks (`cuLaunchHostFunc`, `cuStreamAddCallback`) | yes | run in the client after the stream drains |
| pointer attributes and address ranges | yes | from the client's own allocation table |
| threads | yes | calls from one process are serialized |
| `fork()` after CUDA init | yes | the child opens its own session; not possible with native CUDA |

## Not supported

These calls return `CUDA_ERROR_NOT_SUPPORTED` and log their name once.

- managed memory (`cuMemAllocManaged`) and mapped pinned memory (`cuMemHostGetDevicePointer`)
- IPC handles, peer access, more than one GPU per process
- explicit graph construction (`cuGraphAdd*Node`) and graph exec updates
- arrays, textures, surfaces, mipmaps
- graphics, EGL and VDPAU interop; external memory and semaphores
- virtual memory management (`cuMemCreate`, `cuMemMap`) and memory pools
- stream memory operations (`cuStreamWaitValue*`, `cuStreamWriteValue*`, batch operations)

Run a program with `WARMD_TRACE=1` to see which unsupported calls it makes.

## Behaviour differences

- Large copies go through the session's staging memory: 3 to 5 GB/s instead of 6 to 25 GB/s.
- `cuCtxSetLimit` affects the shared context, so the daemon only allows raising a limit.
- A device fault ends the context for every open session of the daemon, as it does under MPS.
  The other sessions get error 46 and should be retried.
- `cuStreamGetFlags` reports 0 and stream priorities are passed through as hints.
