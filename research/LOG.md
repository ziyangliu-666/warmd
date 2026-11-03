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

## Correction to Result 5

Two stale processes held 995 MiB during Result 5. Clean numbers are in `results/h6b_park_clean.jsonl`: 8199 MiB mapped, 1041 MiB evicted. The 1041 MiB is two contexts (broker and client). Per-process deltas and timings are unchanged. Process: kill stragglers and check that memory is 0 MiB before each run.

## Result 6: long-tail replay, first pass (measured, `results/h7.jsonl`)

Synthetic trace: 932 requests in 300 s over twenty 1.6 GB models, with five resident. There is no real production trace, so this does not measure real idle share.

| design | isolated | miss rate | swap mean | p50 | p99 |
|---|---|---:|---:|---:|---:|
| in-process LRU, native | no | 25.6% | 129 ms | 11.8 ms | 259 ms |
| in-process LRU, warmd | no | 25.6% | 325 ms | 99.6 ms | 967 ms |
| one process per model, device-memory files | yes | 25.6% | 311 ms | 68.1 ms | 878 ms |
| simulated scale-to-zero, favoured baseline | yes | 25.6% | n/a | 2003 ms | 3890 ms |

- Errors of mine: the first simulation let cold starts block the GPU queue, and H4 reported a 193 ms service time that included warm-up. The table uses the fixed simulation (`results/h7_sim.jsonl`).
- H4 is not falsified: 15 times better p99 than the favoured baseline, on a synthetic trace.
- The device-memory design was slow because it copied from pageable memory. It also kept 4.6 GB of allocator cache in idle processes. Both are fixed for the next run.
- 20 idle contexts cost 11135 MiB, which is 27% of the GPU.

## Result 7: properties beyond the happy path (measured, `results/h8.jsonl`)

| test | result |
|---|---|
| pin a read-only page-cache mapping | fails with error 1, so the host tier needs a pinned copy; falsified my zero-copy idea |
| first load with a pinned copy, 7.5 GB | 7.6 s, against 1.0 s without pinning; this is a new cost |
| fork after import to first result | 0.57 to 0.58 s (1.6 GB), 0.64 to 0.66 s (7.5 GB), weights resident; native full start is 4.35 s for 7.5 GB |
| new device memory after another process wrote and freed | 0 nonzero values, in the same process and in a new one |
| cuda-checkpoint on a parked process | frees 532 MiB; checkpoint 376 ms, restore 235 ms, then resume and forward 6 ms, same output |
| seven architectures on read-only weights | no faults; generate output is identical for GPT-2 and Llama |

Open: Whisper and ResNet outputs differ slightly (3e-6, 6e-4). A control run is in progress.

## Result 8: fixes and asynchronous replay (measured, `results/h7b.jsonl`, `results/h11.jsonl`)

| design | isolated | swap mean | p50 | p99 | mean device memory | all idle |
|---|---|---:|---:|---:|---:|---:|
| in-process LRU, native | no | 128.5 ms | 13.4 ms | 244.8 ms | not sampled | not measured |
| process per model, serial router | yes | 117.6 ms | 11.6 ms | 220.9 ms | not sampled | 11139 MiB |
| process per model, async router | yes | 127.9 ms | 11.3 ms | 218.8 ms | 19832 MiB | 11139 MiB |
| async, hibernate after 10 s idle | yes | 288.0 ms | 11.4 ms | 639.8 ms | 14289 MiB | 501 MiB |
| async, hibernate at once | yes | 625.8 ms | 11.9 ms | 1641.8 ms | 11300 MiB | 501 MiB |

All runs: 932 requests. The hibernating and async runs returned 0 wrong outputs.

## Result 9: output differences (measured, `results/h10.jsonl`, `results/h8_zoo_aligned.jsonl`)

Control runs on ordinary allocations show no difference, so the mapped path caused it. At 8-byte tensor alignment, ResNet differs by 6e-4 and Whisper by 3e-6. At 64 bytes and 4 KiB both differ by 0. The weights were byte-equal in every case. The broker now uses a 4 KiB layout, and all seven architectures differ by exactly 0.

## Result 10: hibernation limits (measured, `results/h9.jsonl`, `results/h10.jsonl`)

- Restore rate: 2.7 to 5.7 per second for 1 to 32 processes, sequential or parallel. Hibernation does not escape the driver's rate limit.
- A checkpoint or restore does not stall kernels in another context: p99 is 0.21 ms in both cases.
- Host memory, PSS: 943 MiB parked, 1412 MiB hibernated, 669 MiB parked after a zygote fork. Device memory returns, and host memory pays 469 MiB.

## Result 11: fair lease (measured, `results/h10_lease.jsonl`)

A first attempt with `mp.Lock` was invalid: one process kept the lock. A p99 of 0.08 ms exposed it. Strict round robin, 8 processes, 256 by 256 matmul:

| schedule | ops/s | p50 | p99 |
|---|---:|---:|---:|
| driver time slicing | 5756 | 1.39 ms | 3.38 ms |
| lease 0.2 ms | 3918 | 2.01 ms | 2.46 ms |
| lease 2 ms | 15203 | 0.06 ms | 14.77 ms |
| lease 10 ms | 16694 | 0.06 ms | 0.09 ms; a wait near 70 ms is below the p99 sample |
| warmd, shared context | 22980 | 0.21 ms | 1.89 ms |

Computed from the 0.2 ms row: one context switch costs about 200 µs. For 4096 by 4096 matmuls, a lease gives 108 ops/s against 95 for time slicing, and warmd gives 105.

## Result 12: H2, gVisor boundary (measured in software emulation, no GPU, `results/h2.txt`)

H2 is partly falsified. With `--host-uds=open`, connect, SCM_RIGHTS, mmap and shared-memory polling all work. Futex wakes do not cross the boundary: the test stopped after 6 timeouts of 200 ms. Without the flag, connect is refused. warmd needs a different wake path to run in gVisor. Latency numbers from emulation mean nothing. Device-memory files under gVisor remain untested, because the VM has no GPU.

## Result 13: segments in a shared context (measured, `results/h12.jsonl`, `results/h12c.jsonl`)

A copy of warmd with persistent segments (`scripts/make_warmd_seg.sh`). One new process per task, 7.5 GB model, every output checked.

| design | concurrency | tasks/s | p50 | wrong |
|---|---:|---:|---:|---:|
| warmd today, copies weights | 1 | 0.53 | 1890 ms | 0 |
| warmd today, copies weights | 4 | 1.45 | 2850 ms | 0 |
| warmd with segment | 1 | 17.88 | 54 ms | 0 |
| warmd with segment | 4 | 56.53 | 70 ms | 0 |
| warmd with segment | 8 | 78.87 | 99 ms | 0 |
| native context with file | 1 | 1.25 | 810 ms | 0 |
| native context with file | 8 | 3.76 | 2046 ms | 0 |

Failure kept for the record: the first segment run gave NaN in every task at concurrency 2 or more, with no error. A live block in the zygote's PyTorch allocator was inherited by each fork, so children placed scratch at the same device address. The fix: the zygote holds no allocator memory. A shared context turns this mistake into silent wrong answers.

## Result 14: revocation (measured, `results/h13.jsonl`)

| action on a client that does not park | result |
|---|---|
| broker releases the file | memory stays at 8213 MiB, and client output stays correct |
| cuda-checkpoint with a live import | memory falls to 513 MiB in 3290 ms |
| restore after that | fails with return code 1 after 20 s; broker had already released the file, so the cause is not separated |
| kill the client | memory returns within 24 ms |

So the safe forced reclaim is to kill the client. Park before any checkpoint.

## Result 15: second uid (measured, `results/h14_uid.txt`)

uid 65534 mapped a segment from a root broker: output 27.136173248291016, 12.08 ms per forward pass. The device nodes here are world-writable. Cross-container sharing is still untested, because this container cannot create namespaces.

## Result 16: real checkpoint and real engine (measured, `results/h15.jsonl`, `results/h15_hf.jsonl`)

Qwen2.5-0.5B, a real checkpoint, downloaded.

| test | result |
|---|---|
| transformers, mapped bfloat16 weights | logits differ by 0; generated text identical; 2.69 s against 2.70 s |
| vLLM 0.6.6, weights replaced by a segment | text identical; 0.750 s against 0.741 s |
| vLLM after park, evict and resume | text identical; resume with reload 45.8 ms |
| vLLM memory: ordinary, mapped with broker, parked and evicted | 5595, 5823, 4879 MiB |

- Two false alarms were my tests: the native path first ran in float32, and `from_config` does not read `generation_config.json`. Comparing logits separated model from generation settings.
- New limit: vLLM keeps its KV pool (3.46 GiB here) while parked. A segment frees weights only. An engine must release its mutable pool to reach the state ladder.

## Result 17: exported fd (measured, `results/h15.jsonl`)

The fd is `/dev/nvidiactl` (195:255). A process imported it after the exporter had released its handle and exited, and read back correct data. So a segment does not depend on the exporter process. This narrows the cross-container question but does not answer it.

## Result 18: real arrival shape (computed, `results/h16_trace.json`)

Azure Functions 2019, day 1, 46412 functions, 909 million invocations. These are CPU functions, so this gives arrival shape only. GPU workloads may differ.

- The top 1% of functions make 91.3% of invocations. 86.7% run under once per minute. The median is 32 per day.
- Idle share of resident time, keep-alive 5 minutes: 78% to 98% for all functions, and 94% to 99.9% for the tail, for service times of 10 s to 0.1 s.
- Cold starts per invocation in the tail: 44.5% at 1 minute keep-alive, 11.1% at 5, 7.2% at 10, 1.4% at 60.

## Result 19: real arrival window replayed on the GPU (measured, `results/h17.jsonl`, `results/h17_sim.jsonl`)

Ten real minutes of Azure Functions day 1, 1024 requests, twenty 1.6 GB models with five resident. Arrivals within a minute are random, because the trace has one-minute resolution. All measured runs report 0 wrong outputs where checked.

| design | isolated | misses | p50 | p99 | mean device memory | all idle |
|---|---|---:|---:|---:|---:|---:|
| in-process LRU, native | no | 76 | 11.0 ms | 147.2 ms | not sampled | not measured |
| process per model, segments | yes | 77 | 11.8 ms | 141.2 ms | 19875 MiB | 11139 MiB |
| hibernate after 60 s idle | yes | 78 | 11.8 ms | 431.0 ms | 13887 MiB | 501 MiB |
| hibernate after 10 s idle | yes | 78 | 11.9 ms | 412.6 ms | 12349 MiB | 501 MiB |
| simulated scale-to-zero, favoured baseline | yes | 77 | 11.0 ms | 3891 ms | n/a | n/a |

The GPU ran inference for under 2% of the window. This is one window with one random sample of functions.

## Result 20: full state ladder on vLLM (measured, `results/h18.jsonl`)

Qwen2.5-0.5B, vLLM 0.6.6, three prompts, 64 tokens. At park the process unmaps weights and drops the KV pool; at wake it rebuilds the pool. The test reaches into vLLM internals, with no request in flight. Output is identical to the original in every state.

| state | idle device memory, with broker | overhead before generate | time to result |
|---|---:|---:|---:|
| running | 5823 MiB | 0 | 0.78 s |
| parked | 2271 MiB | 92 ms | 0.87 s |
| hibernated | 1447 MiB | 662 ms | 1.44 s |
| hibernated, segment evicted | 503 MiB | 719 ms | 1.50 s |
| real cold start, page cache warm | 0 | 21.19 s (computed) | 21.97 s |

Computed: 14.6 times faster to the first result than a real cold start, with 11.6 times less device memory than running. The 503 MiB is the broker's context, which many processes share. Limits: the KV pool's contents are lost, so prefix caching does not survive; the first load used a saved copy of the engine's own weights.

## Blocked: cross-container test

`scripts/xc.c` and `scripts/run_h19.sh` are written and have not run. They cover two containers, a second uid, an owner that exits, a client with no capabilities, and a client in gVisor with nvproxy. On 2026-11-03 the rental failed with `insufficient_credit`: account credit is 0 and balance is -0.018. A100 instance 58531275 showed `exited` when it was destroyed, which fits the credit running out, but this is not confirmed. All A100 results had been copied before that.

## Process lessons

- A surprising good number needs a check before a bad one does. Three of mine were wrong: the 8-byte alignment "noise", the unfair lease, and NaN at 84 tasks/s.
- Verify every output against a reference. It caught the NaN result.
- Check for 0 MiB and no stragglers before each run.
- Separate variables at the first odd result.
