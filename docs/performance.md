# Performance

Setup: A100-SXM4-40GB, driver 570.211.01, 8 vCPUs (AMD EPYC 7B12), PyTorch 2.5.1, Triton 3.1.
Raw data is in [`results/a100-570/`](../results/a100-570/). `bench/run.sh` reproduces it.

## Short PyTorch tasks, one process each

Each task draws its own random 256×256 inputs, runs a matmul, a softmax and a few copies, and
checks its result against the CPU. `bench/torch_e2e.py` and `bench/pool_bench.py`.

| how each task starts | 1 at a time | 8 at a time |
|---|---:|---:|
| native, new `python` process | 0.53/s | 2.3/s |
| native, fork after `import torch` | 2.07/s | 4.5/s |
| warmd, new `python` process | 0.64/s | 4.3/s |
| warmd, fork after `import torch` | 8.1/s | 59/s |
| warmd, fork after CUDA and cuBLAS warm-up | 42/s | 133/s |
| warmd, `ZygotePool`, 2, 4 and 8 zygotes | | 229, 202 and 220/s |

Natively, eight concurrent tasks reach only 4.5/s because contexts are created one at a time.
With warmd, the limit is CPU time for `fork()` and Python work. No task returned a wrong result.

Median latency of one warm-fork task with nothing else running is 24 ms: about 10 ms for
`fork()` of a process that has PyTorch loaded and about 6 ms for the task's own work.

## Driver level

`bench/micro.c`, built once against the real driver and once against the shim.

| | native | warmd |
|---|---:|---:|
| `cuInit` plus primary context | 162 ms | 1.9 ms |
| `cuStreamSynchronize` on an idle stream | 0.2 µs | 13.5 µs (p99 33 µs) |
| empty kernel launch plus synchronize | 9.0 µs | 30.4 µs |
| kernel launches without synchronize | 363,000/s | 64,000/s |

Copy bandwidth in GB/s (host to device / device to host):

| size | native pageable | native pinned async | warmd pageable | warmd pinned async |
|---|---|---|---|---|
| 4 KB | 0.42 / 0.36 | 1.24 / 1.61 | 0.15 / 0.17 | 0.13 / 0.17 |
| 2 MB | 11.2 / 4.6 | 24.1 / 12.6 | 7.0 / 3.5 | 10.2 / 3.7 |
| 128 MB | 9.0 / 6.3 | 24.8 / 12.6 | 4.9 / 3.0 | 4.6 / 2.9 |

## Costs inside a task

| step | native | warmd |
|---|---:|---:|
| CUDA init and context | 162 ms | 2 ms |
| first PyTorch host-to-device copy (lazy init, allocator) | 160 to 200 ms | 13 to 31 ms |
| first cuBLAS matmul (handle creation, heuristics) | about 70 ms | about 55 ms of CPU work |
| fork of a warmed process | not possible | 7 to 12 ms |

## Known costs

- Every call into the daemon costs 10 to 15 µs, so code that launches many tiny kernels is
  slower than native. Calls that only enqueue work on a stream (launch, memset, async copy,
  event record) could return without waiting for the daemon; that is planned.
- Copies are staged through session memory, which adds a CPU copy. Planned: pinned memory shared
  with the daemon so pinned copies go straight to the GPU, and double-buffered staging for
  pageable copies.
- The first cuBLAS or cuDNN call still costs about 55 ms of CPU time in each new process.
  `ZygotePool` pays it once per zygote.
- After a fault, the GPU needs 0.4 to 0.6 s to report it. Promotion of the standby takes
  microseconds. The first task on the new worker reloads GPU code (50 to 100 ms for PyTorch).
