# Performance

A100-SXM4-40GB, driver 570.211.01, 8 vCPUs (EPYC 7B12), PyTorch 2.5.1. Raw data:
`tests/bench/results/a100-570/`. Reproduce with `tests/bench/run.sh`.

## Short PyTorch tasks

Each task is its own process. It makes two random 256×256 matrices from its own seed, runs a
matmul and a softmax, copies results back, and checks them against the CPU.

| how the process starts | 1 at a time | 8 at a time |
|---|---:|---:|
| native, new python | 0.53/s | 2.3/s |
| native, fork after `import torch` | 2.07/s | 4.5/s |
| warmd, new python | 0.64/s | 4.3/s |
| warmd, fork after `import torch` | 8.1/s | 59/s |
| warmd, fork after CUDA and cuBLAS warm-up | 42/s | 133/s |
| warmd, `ZygotePool` with 2, 4, 8 zygotes | | 229, 202, 220/s |

The native rows are limited by context creation, the warmd rows by CPU. A single warm-fork task
takes 24 ms: about 10 ms to fork a process with PyTorch loaded and about 6 ms of work.

## Driver calls

| | native | warmd |
|---|---:|---:|
| `cuInit` and primary context | 162 ms | 1.9 ms |
| `cuStreamSynchronize`, idle stream | 0.2 µs | 13.5 µs |
| empty kernel launch and synchronize | 9.0 µs | 30.4 µs |
| launches per second, no synchronize | 363,000 | 64,000 |

## Copies

GB/s, host to device / device to host.

| size | native pageable | native pinned | warmd pageable | warmd pinned |
|---|---|---|---|---|
| 4 KB | 0.42 / 0.36 | 1.24 / 1.61 | 0.15 / 0.17 | 0.13 / 0.17 |
| 2 MB | 11.2 / 4.6 | 24.1 / 12.6 | 7.0 / 3.5 | 10.2 / 3.7 |
| 128 MB | 9.0 / 6.3 | 24.8 / 12.6 | 4.9 / 3.0 | 4.6 / 2.9 |

Copies go through session memory, which costs an extra CPU copy. Programs that launch many tiny
kernels also pay the per-call cost.

