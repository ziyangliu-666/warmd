# Research log: abstractions for elastic GPU computing

Started 2026-11-02. Hardware: one rented A100-SXM4-40GB, driver 570.211.01, 8 vCPUs, 62 GB RAM.
Scripts are in `scripts/`. Raw results are in `results/`.

## Method

- State each hypothesis and its falsifier before the run.
- Keep failures in this log.
- Label every number as measured, computed, or taken from a source.

## Open hypotheses

| id | hypothesis | falsified if |
|---|---|---|
| H1 | Context creation is a small part of a real cold start, and warmd slows weight loading | context share above 10%, or warmd total is shorter |
| H2 | warmd's primitives do not cross a gVisor boundary | the client works in the sandbox |
| H3 | Device memory can be shared across contexts, with better fault isolation than one context | IPC fails, or a fault reaches other contexts |
| H4 | A device-memory cache improves long-tail p99 by a large factor | p99 improves by less than 5 times |
| H5 | warmd copies can approach native speed | the extra CPU copy is not the bottleneck |
| H6 | The public VMM API gives read-only, zero-copy weight sharing across native contexts, so a device-memory file with mmap semantics replaces driver interposition | the import fails, the read-only mapping is writable, or a fault reaches another context |

## Failures so far

- gVisor cannot start in the rented container: it has no namespace permission. H2 moved to a QEMU VM.

## Result 1: H1, cold-start stages (measured, `results/h1.jsonl`)

Median of 3 runs, first run per configuration dropped, seconds. Page cache warm, Llama graph with random weights.

| size | impl | import torch + libs | context | weights to GPU | first inference | second inference | total |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.6 GB | native | 3.14 | 0.196 | 0.192 | 0.324 | 0.003 | 3.88 |
| 1.6 GB | warmd | 3.30 | 0.029 | 0.342 | 0.149 | 0.006 | 3.85 |
| 7.5 GB | native | 2.98 | 0.199 | 0.762 | 0.345 | 0.012 | 4.35 |
| 7.5 GB | warmd | 3.00 | 0.031 | 1.586 | 0.154 | 0.024 | 4.85 |
| 15.6 GB | native | 3.02 | 0.201 | 1.818 | 0.358 | 0.024 | 5.51 |
| 15.6 GB | warmd | 3.02 | 0.036 | 3.322 | 0.189 | 0.051 | 6.78 |

- H1 is not falsified. Context creation is 3.6% to 5% of the native total, and warmd is equal or slower.
- My earlier estimate of a 5 times copy penalty was too high. `.to("cuda")` from pageable memory shows about 2 times.
- New finding: warmd doubles steady-state inference time on this workload (12 ms to 24 ms). The per-call cost is not small for a model with many short kernels.
- warmd halves the first inference (0.345 s to 0.154 s) because the worker keeps GPU code loaded.
- Invalid data: the disk-cold warmd rows ran after the native rows and were faster by up to 5.5 times. Eviction with `posix_fadvise` was probably partial. Do not compare them. The native cold rows show that disk reads can dominate (7.5 GB: 23 s).

## Result 2: H6, device-memory file, copy path only (measured, native C, `scripts/h6_vramfile.c`)

- Import 0.07 ms, map 0.45 ms, device memory delta 2 to 4 MiB for an 8 GB file.
- A copy write through the read-only mapping returns error 1, and no other process is affected.
- The memory survives reader exit. Kernel writes are not tested yet.

## Result 3: H3, one context per process against one shared context (measured)

Isolation, `results/h3_fault.jsonl`. The first run was invalid: a Triton import error looked like a fault (`results/h3_invalid_fault_tests.jsonl`).

| test | native contexts | warmd |
|---|---|---|
| store to another process's device address | writer faults, victim unchanged | store completes, victim value changes from 0 to 9 |
| wild store | bystander and owner keep working | documented: all sessions end |

Concurrency, `results/h3.jsonl`, aggregate matmuls per second:

| size | procs | native | warmd |
|---|---:|---:|---:|
| 256 | 1 | 17383 | 6252 |
| 256 | 2 | 5726 | 9540 |
| 256 | 8 | 5698 | 22980 |
| 4096 | 1 | 112 | 109 |
| 4096 | 8 | 94 | 105 |

- Time slicing between contexts costs 67% on short kernels. warmd's shared context wins by 4 times at 8 processes. This is warmd's real advantage, and it is the one thing separate contexts cannot give.
- One native process is 2.8 times faster than warmd on short kernels.
- A native context costs about 497 MiB of device memory.

## Result 4: H5, copy path (measured, `results/h5.jsonl`)

- Registered shared memory, zero copy: 24.7 GB/s, equal to native pinned.
- Staged like warmd today (8 MiB, one thread): 4.6 GB/s. The memcpy (6.1 GB/s) is the limit.
- Staged with 256 MiB and four threads: 14.9 GB/s.
- H5 is not falsified. A direct path through registered memory removes the penalty.

## Result 5: H6, device-memory files on PyTorch (measured, `results/h6b.jsonl`, 7.5 GB model)

`scripts/vf.py` implements a broker that owns physical memory, and clients that map it read-only at a stable address. Clients use the unmodified NVIDIA driver library.

| property | measured |
|---|---|
| output against ordinary weights | identical sum, 27.136173248291016 |
| forward pass | 11.74 ms mapped, 11.87 ms native, about 24 ms warmd (Result 1) |
| first open, including load from page cache | 0.997 s |
| park (unmap) | 0.44 ms |
| resume, weights resident | 1.46 ms |
| evict | 5.14 ms; used memory falls from 9190 to 2032 MiB |
| resume after evict | 1009 ms, then first forward 13.3 ms with no warm-up |
| kernel store into read-only weights | writer faults, report 919 ms; bystander output unchanged; weight unchanged |
| 1, 4, 8, 16 processes on one file | 9190, 10817, 12983, 17317 MiB; about 542 MiB per extra process |

- H6 is not falsified. My earlier claim that read-only sharing might need kernel changes was wrong: the public VMM API does it.
- Failure kept for the record: the first loader took 219 s because Python slice assignment on a ctypes array is slow.

## Current model of the design space

The result suggests that two things warmd joined should be separate:

- Weights are immutable data. They do not need a shared context, a shim, or a process that stays alive.
- A context is a trust domain and a time-slice unit. Share one only when processes trust each other and run short kernels at the same time.

Open: 2032 MiB remains after eviction with one client and one broker, so a parked process costs more than a bare context. Find out what that is.
