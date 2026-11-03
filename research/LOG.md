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

## Result 21: capacity model on real arrivals (computed, `results/h20_fleet.jsonl`, `scripts/fleet_model.py`)

This is a model, not a measurement. Azure Functions days 1 and 2, leaving out the top 1% of functions, keep-alive 5 minutes. Target GPU utilization 70%, 38 GB usable. Dedicated means one GPU per resident model.

| service time per request | model size | dedicated, peak GPUs | segments, peak GPUs | peak saving, day 1 and day 2 |
|---:|---:|---:|---:|---:|
| 0.1 s | 1.6 GB | 18664, 19014 | 786, 801 | 23.8, 23.7 times |
| 0.1 s | 8 GB | 18664, 19014 | 3929, 4003 | 4.8, 4.8 times |
| 0.1 s | 16 GB | 18664, 19014 | 7859, 8006 | 2.4, 2.4 times |
| 1 s | 1.6 GB | 18664, 19014 | 2379, 3362 | 7.8, 5.7 times |
| 1 s | 8 GB | 18664, 19014 | 3929, 4003 | 4.8, 4.8 times |
| 10 s | any | 18664, 19014 | 23791, 33615 | none: 0.8, 0.6 times |

- The result can go against the claim. At 10 s per request, compute is the limit and sharing saves nothing. The dedicated figure there also ignores queueing, so it is too low. Segments cannot create compute.
- The saving is the ratio of device memory to model size wherever memory is the limit. The arrival shape sets only how many models are resident.
- The trace is CPU functions. Real GPU request sizes and rates are unknown, and they decide which row applies.

## Result 22: source reading, predictions for the cross-container test

These are predictions from source, not results. Sources: NVIDIA open kernel modules 615.78.08, while our runs used 570; gVisor at HEAD on 2026-11-03.

- `rmobjexportimport.c` says that another RM client can import an exported handle "even if the source rm client gets destroyed". This agrees with Result 17.
- Export handles live in one kernel-wide map. The import path in `os.c` checks that the fd is the control device (`nv_get_file_private`, minor number) and that its handle is valid. I found no uid, pid or namespace check in the code I read. I did not read every function it calls.
- gVisor's `ctrlHasFrontendFD` looks the fd up in the sandbox's own table, and it returns EINVAL unless the file is a `frontendFD`. An fd from outside the sandbox has a different type.

Predictions for `run_h19.sh`, written before any run:

| case | prediction |
|---|---|
| A, no containers | pass |
| B, two containers | pass |
| C, another uid | pass |
| D, owner exits after sending | pass |
| E, no capabilities, read-only root | pass |
| F, client in gVisor, owner outside | fail with an invalid-value error from nvproxy |

If F fails as predicted, sharing across gVisor sandboxes needs a change in nvproxy. Possession of the fd is the capability, so a broker must control who receives it. The forum report of failure across containers was on CUDA 11.3 and did not reach a cause.

## Result 23: cross-container sharing (measured, `results/h19.txt`, `results/h19b.txt`)

RTX 3060 in a VM, driver 535.183.01, a second GPU model and driver branch. 1 GiB segment. All predictions from Result 22 held.

| case | predicted | result |
|---|---|---|
| A, no containers | pass | pass; import 0.07 ms, map 0.17 ms |
| B, two containers, different pid namespaces | pass | pass; import 0.07 ms, map 0.18 ms |
| C, client uid 65534 | pass | pass |
| D, owner container exited before the import | pass | pass |
| E, client with no capabilities and read-only root | pass | pass |
| H, client in an unprivileged user namespace | none | pass |
| G1, owner and client in one gVisor sandbox (control) | none | pass; import 0.44 ms |
| F, owner outside, client in gVisor | fail | fail with CUDA error 101; the fd shows as `host:[4]` |
| F2, owner in gVisor, client outside | none | invalid test: the socket stays inside the sandbox |

The earlier forum failure across containers does not reproduce here. Cause unknown. The error it reported, invalid device, is the same code that case F returns.

## Result 24: a nvproxy change for gVisor (measured, `results/h21b.txt`, `results/h21_patch.diff`)

`scripts/gvisor_nvproxy_host_fd.py` makes `ctrlHasFrontendFD` accept a host fd: 9 lines added, 3 removed, in one file.

| case | result |
|---|---|
| F, patched | pass; import 0.86 ms, map 0.58 ms, delta 0 MiB, write rejected |
| F, patched, second sandbox | pass |
| F, unpatched, same setup | fail with error 101 |
| ordinary file, patched and runc | rejected with error 101 |

Limits: built from the go branch of 2026-11-02, with driver version forced to 535.183.06, because 535.183.01 is not on the supported list. No security review. gVisor maintainers have not seen it.

## Result 25: real kernels across boundaries (measured, `results/h22.txt`, `results/h23.txt`)

The broker runs in one runc container, 1.6 GB Llama graph with random weights.

| client | output | forward pass |
|---|---|---:|
| runc, ordinary weights | -10.301668167114258 | 6.68 ms |
| runc, segment | -10.301668167114258 | 6.64 ms |
| gVisor patched, segment | -10.301668167114258 | 7.86 ms |

A real kernel (`fill_`) stores into read-only weights. In all four combinations of runc and gVisor for writer and bystander, the writer gets `illegal memory access`, its store to ordinary memory works, and the bystander's output and weights do not change.

Failure kept for the record: the first fault test reported "faulted", but the message was "Failed to find C compiler". Triton never ran. Read the text of a pass condition, not only its presence.

Not tested in gVisor: hibernation with cuda-checkpoint, park and evict under load, and vLLM.

Spend: the VM ran 0.41 hours at about $0.09 per hour, about $0.04 by estimate. Account credit went from 9.98 to 9.95.

## Result 26: which capacity row fits LLM traffic (computed, `results/h26_llm.json`, `scripts/llm_service.py`)

Public Azure LLM inference traces, about one hour each. Earlier I said GPU request data was out of reach. That was wrong: public data exists.

| trace | requests/s | context tokens, median | generated tokens, median |
|---|---:|---:|---:|
| conversation | 5.53 | 1020 | 129 |
| code | 2.57 | 1469 | 13 |

Assumed batched rates, not measured: prefill 2000 to 40000 tokens/s, decode 300 to 6000 tokens/s. That gives 0.06 to 1.28 GPU seconds per request, so the model is limited by memory in nearly every case. Only 1.6 GB models at the lowest rates are limited by compute.

| model size | saving, memory-limited |
|---:|---:|
| 1.6 GB | 23.8 times |
| 8 GB | 4.8 times |
| 16 GB | 2.4 times |
| 40 GB | 1.0 times: none |

- The saving is usable device memory divided by model size. It comes from packing.
- A single-process LRU would pack the same way, without isolation. So segments do not create this saving. They make it available between tenants who do not trust each other.
- A model that fills a GPU gains nothing here. Frontier models are out of scope for this saving.
- KV memory for active requests is not in the model, so real savings are lower.
- These traces show large services, not the long tail, and arrivals in the model still come from CPU functions. No public trace links many small GPU models to their arrivals.

## Result 27: what a context costs, and how to cut it (measured, `results/h28.jsonl`, `results/h28b.jsonl`)

RTX 3060, driver 570.195.03 in a VM. Bare start in C, then PyTorch in a container.

| setting | cuInit | context | device memory | parallel starts per second, 16 processes |
|---|---:|---:|---:|---:|
| no persistence mode | 562 ms | 160 ms | 109 MiB | 4.01 |
| no persistence, one connection | 575 ms | 57 ms | 109 MiB | 8.67 |
| persistence mode | 36 ms | 197 ms | 109 MiB | 4.81 |
| persistence, one connection | 35 ms | 86 ms | 109 MiB | 14.16 |
| persistence and MPS | 13 ms | 133 ms | 102 MiB | 7.53 |
| persistence, MPS, one connection | 13 ms | 88 ms | 102 MiB | 13.10 |

"One connection" is `CUDA_DEVICE_MAX_CONNECTIONS=1`, so the driver creates one hardware queue.

- My MPS result first looked like a 47 times faster cuInit. The persistence control shows that persistence mode alone gives most of it. MPS is not needed for that.
- One connection is the largest single gain: the start rate rises 2.9 times, and warmd was built for the old rate. The cost: streams share one queue, which can add false dependencies. Not measured.
- No setting reduced device memory. PyTorch adds about 41 MiB, for about 150 MiB per process.
- Bare context memory is 109 MiB here and about 497 MiB per PyTorch process on the A100. Both are about 1.2% of device memory. Two points do not show a law.

## Result 28: MPS on the context axis (measured, `results/h28b.jsonl`, `results/h29.jsonl`)

Same GPU. 256 by 256 matmuls, operations per second:

| design | 1 process | 8 processes | p99, 8 processes |
|---|---:|---:|---:|
| native contexts | 14761 | 11667 | 0.74 ms |
| MPS | 10998 | 74803 | 0.13 ms |
| warmd | 3865 | 60385 | 0.18 ms |

- With MPS, a wild store into another process's address faults, so address spaces stay separate. But the fault killed the victim and the bystander: MPS shares one fault domain, like warmd.
- Segments work under MPS: same output, park 0.43 ms, resume 1.19 ms.
- So MPS replaces warmd for many short kernels in one trust domain, with no driver shim. warmd keeps fork after CUDA init.

## Result 29: the ladder in gVisor (measured, `results/h29.jsonl`, `results/h30.jsonl`)

Broker in runc, patched gVisor, 1.6 GB model, same output in every step.

| step | runc | gVisor |
|---|---:|---:|
| forward pass | 6.68 ms | 8.01 ms |
| park | 0.46 ms | 2.18 ms |
| resume, resident | 1.13 ms | 3.91 ms |
| reload after evict | 140.4 ms | 148.0 ms |
| first forward after reload | 7.46 ms | 11.94 ms |
| cuda-checkpoint | 180 ms | 390 ms |
| restore | 229 ms; 121 ms with one connection | 468 ms |
| fork from a zygote to first result | 0.77 to 0.87 s | 3.73 to 3.92 s |
| importing PyTorch in the zygote | 9.3 s | 42.1 s |

- My prediction from source was wrong: cuda-checkpoint works in gVisor. Reading only nvproxy's own files was not enough.
- Restore rate with 8 processes in runc: 4.99 per second, 11.8 with one connection.
- gVisor makes the process-free states slow, at 3.8 s to a result. Parked and hibernated processes matter more there.
- vLLM did not run on this VM: the install pulled incompatible versions of transformers, tokenizers and huggingface-hub. The user said to drop it.

Spend: this VM ran 1.36 hours, about $0.13 by estimate. Credit went from 9.95 to 9.82.

## Result 30: security review of the nvproxy change (source reading, `NVPROXY_REVIEW.md`)

The driver checks only the device number of the fd, then trusts the file's private data. Version 1 of the change let any host-backed file reach that check, which exposes a type confusion if the host gives the sandbox a block device numbered 195:255. Version 2 checks with fstat for the control character device first. It compiles for linux/amd64 (`results/h31_patch_v2.diff`). It has not run on a GPU.

## Result 31: one hardware queue on an A100 (measured, `results/h32.jsonl`)

A100-SXM4-40GB, driver 580.178.04, persistence mode on. Three rounds of each start test.

| measure | default | one connection |
|---|---:|---:|
| parallel starts, 16 processes | 4.34, 4.41, 4.36 per s | 12.37, 12.30, 12.40 per s |
| context creation | 135 to 165 ms | 76 to 88 ms |
| bare context device memory | 427 MiB | 426 MiB |
| parallel restore, 8 processes | 5.86 per s | 10.69 per s |
| sequential restore | 3.28 per s | 4.71 per s |
| single-stream forward, 1.6 GB model | 2.83 ms | 2.88 ms |
| 128 by 128 matmuls, 4 or 8 streams | 56448, 56597 ops/s | 60149, 60608 ops/s |
| 512 by 512 matmuls, 4 or 8 streams | 46069, 46016 ops/s | 38229, 38421 ops/s |

- The RTX 3060 result holds on the A100: starts rise 2.8 times and restores 1.8 times.
- The cost is real: when kernels from several streams could overlap, one queue loses 17%. Single-stream work does not change. Use it per process, for inference processes with one stream.
- Bare context memory is 427 MiB here and 109 MiB on the RTX 3060, about 1.0% and 0.9% of device memory. No setting changes it.

Spend: about $0.06.

## Result 32: version 2 of the nvproxy change on hardware (measured, `results/h33.txt`, `results/h33_host_fds.txt`)

RTX 3060 VM, driver 570.195.03, both versions built from the same gVisor commit.

| case | version 1 | version 2 |
|---|---|---|
| owner in runc, client in gVisor | pass; import 0.70 ms | pass; import 0.60 ms |
| owner and client in one sandbox | pass | pass |
| host fd of `/dev/null`, `/dev/nvidia0`, a fresh `/dev/nvidiactl`, a regular file, sent into the sandbox | all rejected, error 101 | all rejected, error 101 |

- Version 2 keeps the working case and rejects every host fd tried.
- The tests cannot tell the two versions apart. The only case that version 2 adds protection for, a block device numbered 195:255, cannot be opened without a driver behind it, so it was not tested. Its value is defence in depth.
- A fresh `/dev/nvidiactl` is rejected because it holds no exported handle.
- The kernel log showed no NVIDIA messages during the negative cases.

Spend for this VM: 0.23 hours, about $0.02. Account credit is 9.74 after both validation runs, against the approved cap of $3.

## Result 33: sixteen vLLM models on one A100 (measured, `results/h34/`)

Setup:
- Hardware and software: an A100 40 GB, vLLM 0.6.6, and Qwen2.5-1.5B-Instruct in bf16. The weights are 2.89 GiB.
- Models: 16 copies of the model, each with its own segment name.
- Trace: Azure conversation lengths, with Zipf popularity across the 16 models. It has 1481 requests in 600 s.
- Limits: at most 6 models are active at a time. Each model has the same KV pool, `gpu_memory_utilization` 0.12.
- `plain`: a request for an inactive model stops the least recently used idle engine and cold-starts a new one. One engine changes at a time.
- `seg`: all 16 processes stay alive. An idle model parks: its weights are unmapped and its KV pool is dropped. After 2 s it is checkpointed with cuda-checkpoint. The broker budget is 18 GiB, which holds six resident segments.

| | plain | seg | seg, broker killed at 150 s |
|---|---|---|---|
| requests completed, errors | 1481, 0 | 1481, 0 | 745 of 745, 0 |
| tokens per second | 251.3 | 363.2 | 343.9 |
| latency p50, p90, p99 | 171, 306, 424 s | 30, 105, 132 s | 31, 79, 108 s |
| model switches | 40 cold starts, mean 20.0 s | 89 wakes, mean 4.26 s | 49 wakes, mean 4.74 s |
| device memory mean, max | 22.0, 24.6 GiB | 27.1, 29.3 GiB | 28.2, 31.6 GiB |

- With the same limit on active models, seg serves 1.45 times the tokens. Its p50 is 5.6 times lower and its p99 is 3.2 times lower.
- Both modes are overloaded, so most of the latency is queueing. The KV pool holds only about 8000 tokens per model, and vLLM preempted requests in both modes. The ratio matters more than the absolute latency.
- seg used 5 GiB more device memory on average. Each parked process that is not yet checkpointed keeps its context, and the broker keeps its own context. The comparison is therefore not at equal memory. With six more resident segments in the 5 GiB, plain could keep about one more engine.
- A wake costs 4.26 s, not milliseconds. The ladder predicted about 0.25 s for a restore plus milliseconds for a remap. I did not measure the parts. The candidates are the cuda-checkpoint restore of a vLLM process, a wake that waits for a checkpoint still in progress, and a segment reload after an eviction. This is the next thing to measure: if the wake falls to under 1 s, seg gains much more.
- Killing the broker at 150 s caused no errors and no failed requests. Mapped segments stayed valid, and later opens went to the new broker.

Four harness bugs made the first three attempts invalid. Each one is fixed in `scripts/h34.py`:
- The broker's budget check did not count a load in progress. Two loads at the same time both passed, and the broker went past its budget. This was a real bug in `segments.py`. It is fixed, and a test covers it.
- Two vLLM cold starts at the same time sized their KV pools from each other's free memory, and they failed.
- At setup, the six active models filled the budget, so the seventh model could not open its segment.
- A model that another thread was parking or stopping could still take a request. Two threads then used one pipe.

Spend: the A100 at $0.337 per hour. Credit went from 9.73 to 7.68, so $2.05 for this round with the GPU tests and setup, against the approved cap of $3. The instance is destroyed.

## Result 34: what a vLLM wake costs, and seg against plain at about equal memory (measured, `results/h34d/`)

The setup is the same as in Result 33, on the same host. Two settings changed for all three runs:
- The CPU swap space per vLLM process is 2 GiB, not 4. With 4 GiB, 16 processes pin 64 GiB of host memory.
- The segment file is in `/dev/shm`.

| | seg, hibernate after 2 s | seg, hibernate after 30 s | plain, 7 engines |
|---|---|---|---|
| tokens per second | 363.7 | 362.2 | 266.2 |
| latency p50, p90, p99 | 33, 123, 245 s | 29, 95, 126 s | 148, 297, 429 s |
| device memory mean, max | 27.1, 29.3 GiB | 30.6, 32.7 GiB | 25.6, 29.8 GiB |
| switches | 91 wakes, 91 restores | 97 wakes, 64 restores | 43 cold starts, mean 18.4 s |

The parts of a wake, as the mean, with the p50 in brackets:

| part | hibernate after 2 s | hibernate after 30 s |
|---|---|---|
| wait for a checkpoint in progress | 0.02 s (0.00) | 0.01 s (0.00) |
| cuda-checkpoint restore | 0.98 s (0.61) | 1.05 s (0.65) |
| map the segment, with a reload from pinned host memory | 0.24 s (0.22) | 0.24 s (0.22) |
| rebuild the KV pool | 2.32 s (2.17) | 2.22 s (2.05) |
| whole wake | 3.56 s | 3.17 s |
| checkpoint, off the request path | 2.41 s (1.66) | 1.51 s (0.86) |

- Most of the wake is the KV pool rebuild. At park, the harness drops the whole vLLM cache engine. A resume then allocates the 2 GiB pinned CPU swap space again, as well as the 0.15 GiB GPU pool. The likely fix is to keep the CPU part and reallocate only the GPU part. If that works, the wake falls to about 1.2 s.
- Every wake reloaded its segment: 91 of 91. The budget holds six segments and six models are active, so a parked segment is always evicted. A reload costs only 0.24 s for 2.89 GiB, so the budget matters less than the KV rebuild.
- A restore costs 0.98 s, not the 0.24 s of the ladder. A vLLM process holds more device state than the test processes did.
- At about equal memory, seg still serves 1.37 times the tokens of plain: plain with 7 engines used 25.6 GiB, and seg used 27.1 GiB. One more plain engine raised throughput from 251 to 266 tokens per second, and p99 did not change.
- If idle models stay parked for 30 s before hibernation, a third of the wakes skip the restore. p99 drops from 245 to 126 s, and device memory rises by 3.5 GiB.
- The same seg configuration had a p99 of 132 s in Result 33 and 245 s here. Throughput agreed within 0.2%. The tail of an overloaded queue varies from run to run. Compare throughput and p50 across runs, and treat a single p99 with care.

Spend: two hosts failed before this one and were destroyed. One had a disk that could not write 3 GB in 10 minutes, and the other did not finish an image pull in 80 minutes. Credit went from 7.68 to 7.18, so $0.50 for the round, against the approved cap of $1.5.

## Result 35: keep the host part of the KV pool across a park (measured, `results/h34e/`)

The fix comes from Result 34. At park, the harness now drops only the GPU KV tensors. The pinned CPU swap space stays allocated. At resume, it allocates only the GPU tensors again. vLLM 0.6.6 fills both parts with `torch.zeros`, and the CPU part is 2 GiB of pinned memory. Everything else is the same as in Result 34, on the same host.

| | seg, 2 s, Result 34 | seg, 2 s, kept | seg, 30 s, kept | plain, 7 engines, Result 34 |
|---|---|---|---|---|
| tokens per second | 363.7 | 376.2 | 376.2 | 266.2 |
| latency p50, p90, p99 | 33, 123, 245 s | 15, 69, 101 s | 15, 54, 90 s | 148, 297, 429 s |
| mean wake | 3.56 s | 0.96 s | 0.56 s | cold start 18.4 s |
| KV rebuild in a wake | 2.32 s | 0.008 s | 0.006 s | |
| restore, segment map | 0.98, 0.24 s | 0.71, 0.23 s | 0.69, 0.23 s | |
| device memory mean | 27.1 GiB | 27.1 GiB | 30.8 GiB | 25.6 GiB |

- The KV rebuild fell from 2.32 s to 8 ms, and the mean wake fell from 3.56 s to 0.96 s. The p50 fell from 33 s to 15 s.
- Against plain at about equal memory, seg now has a p50 that is 10 times lower and a p99 that is 4.2 times lower.
- Both seg runs reached the same tokens per second. Every request completed, so this number is the fixed total of tokens divided by the time to drain the trace. It measures how soon the work finishes. It does not measure peak capacity. plain needed 1.41 times as long for the same work.
- The wake is now a restore of about 0.7 s plus a segment reload of about 0.23 s. The reload happens on almost every wake, because the budget holds exactly as many segments as there are active models. A larger budget removes it at the cost of device memory.
- The general rule for park: free only device state, and keep host state. Host memory is the cheap resource, and pinned allocation is slow.

Spend: credit went from 7.18 to 6.90, so $0.28. The round total is $0.78, against the approved cap of $1.5. The instance is destroyed.

## Process lessons

- Time each part of a slow step before you guess. I predicted that the restore dominated the wake. The KV rebuild did, and the fix was four lines.
- Rent a host that already has the image cached. A host that ran the image before starts in under a minute. A new host can take more than an hour.

- Check the worker state machine before you spend GPU hours. Three of four H34 attempts failed on harness races, not on the system under test.

- Add the obvious control before crediting a new tool: persistence mode explained most of the MPS start-up gain.
- A prediction from reading part of a code base is weak. cuda-checkpoint in gVisor was predicted to fail and worked.

- A surprising good number needs a check before a bad one does. Three of mine were wrong: the 8-byte alignment "noise", the unfair lease, and NaN at 84 tasks/s.
- Verify every output against a reference. It caught the NaN result.
- Check for 0 MiB and no stragglers before each run.
- Separate variables at the first odd result.
