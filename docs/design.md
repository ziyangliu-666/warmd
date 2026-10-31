# Design

## Problem

Before a short GPU task runs its first useful kernel it pays fixed costs. We measured on an A100
with driver 570: about 1.9 s for the Python stack (`import torch` and friends) and about 190 ms
to create and destroy a CUDA context. The second cost is serial: one GPU completes roughly five
context lifecycles per second. Fresh processes per task therefore cap a GPU at a few tasks per
second, whatever the task size.

warmd moves these costs out of the task but keeps one process per task.

## Components

```
task process                                  warmd daemon (one per GPU)
  application, PyTorch, cudart                  supervisor: socket, sessions, failover, metrics
  libcuda.so.1 = shim   --- unix socket --->      (no CUDA in this process)
                        <-- memfd + eventfd --  active worker: one CUDA context, N threads
                        === shared memory ===>  standby worker(s): context built, idle
```

### Shim

`shim/shim.c` is a `libcuda.so.1` that implements the driver API by calling the daemon.

- cudart 12 looks up nearly every driver function with `cuGetProcAddress`, by base name and
  version (for example `"cuMemAlloc"` at 12040). The shim maps names to its functions and picks
  the right signature where it changed between versions (`cuStreamGetCaptureInfo`,
  `cuGraphInstantiate`).
- cudart and cuBLASLt also call undocumented driver tables through `cuGetExportTable`. Without
  them `cudaGetDeviceCount` already fails. See [hidden-tables.md](hidden-tables.md).
- cuBLASLt looks up a fixed list of about 460 driver symbols with `dlsym` and gives up if one is
  missing. `shim/stubs.c` defines every symbol the real driver exports as a weak stub, so a
  function the shim implements always takes precedence.
- Device attributes are fetched once per process. Contexts are virtual. Device pointers are the
  real addresses in the daemon's context.
- GPU code images are cached in the daemon. An image that lives in a mapped library is keyed by
  file identity and offset; other images by a hash of their content.
- Asynchronous copies use a ring in the session memory. Host-to-device sources are copied into
  the ring at call time. Device-to-host copies into pinned memory land in the ring and are copied
  out at the next point where the program can observe completion.
- After `fork()` the child drops the inherited session and opens a new one on its first call.

### Supervisor

`host/warmd.c` runs a single-threaded epoll loop and never initializes CUDA. It owns the socket,
the pool of session regions (a memfd and an eventfd each), the wait queue, the workers, failover
and metrics. When a client's socket closes, it asks the worker to free the session, then zeroes
the region's resident pages and returns it to the pool.

### Worker

`host/worker.c` is the same binary started with `--worker`. It creates the context, maps and pins
every session region once, and creates one non-blocking stream per region. After promotion, N
threads wait on one epoll set with `EPOLLONESHOT`, so different sessions run in parallel while
each session is served by one thread at a time. A session's default stream is its own stream,
so sessions never wait on each other through the legacy stream; inside a session, legacy
default-stream ordering is reproduced with events. All resources are recorded per session and
freed when it closes. After a sticky device error the worker reports the session and exits.

### ZygotePool

`launcher/warmd/pool.py` starts zygote processes that import PyTorch and warm up CUDA and cuBLAS
through the shim, then fork one child per task. Before forking, a zygote drops its cuBLAS
workspaces and cached allocator blocks so children do not share them. See [pool.md](pool.md).

## Decisions

**Forward the driver API, not the runtime API.** PyTorch, cuBLAS, cuDNN and Triton ship or load
their own copies of runtime code, and all of them end at `libcuda.so.1`.

**One context per GPU, one process per task.** The context is the serial bottleneck. Keeping the
process boundary keeps host-side isolation and crash cleanup simple.

**A supervisor without CUDA, and disposable CUDA workers.** After a sticky fault a context cannot
be repaired: even `cuDevicePrimaryCtxReset` keeps returning error 700. The process holding CUDA
must be replaceable, and the process holding sockets and sessions must survive it.

**A private memfd per session, sent over a unix socket.** A client can only map its own memory.
The same mechanism works across containers without a shared `/dev/shm`.

**eventfd to wake the worker, futex to wake the client, a short spin before sleeping.** A round
trip takes 10 to 15 µs without keeping a CPU busy.

**Fault handling in two layers.** The daemon fails over quickly and returns a sticky, retryable
error. The pool knows about tasks, so it retries them and finds the one that caused the fault.

**Weak stubs for every driver export,** generated from the real driver by `tools/gen_stubs.sh`.

**Allow `fork()` after CUDA init.** Children reconnect; handles inherited from an older worker
generation (after a failover) are detected and the child fails fast.

**Queue clients when full.** A full daemon makes new clients wait. If the wait times out, the
client sees "out of memory", never "context lost", so pools do not mistake it for a fault.

## Alternatives we measured or rejected

- MPS shares the GPU, but each client still creates a context, and a fault in one client can
  affect the others.
- Checkpoint and restore (cuda-checkpoint, CRIUgpu) restored in about 270 ms in our tests, slower
  than a fresh 180 ms init for small tasks, and the context is rebuilt anyway.
- A long-lived worker that runs many users' code avoids the cost but gives up the process
  boundary.
