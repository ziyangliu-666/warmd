# Separate lifetimes: segments, contexts and processes

Status: a research result from one A100 (driver 570) and one RTX 3060 (driver 535), on 2026-11-02 and 2026-11-03. See `LOG.md` for data and failures. It is not a production design.

## The claim

Elastic GPU systems treat GPU state as one thing with one lifetime: the process. Cold start, idle cost, snapshot size and blast radius all follow from that choice.

Three things have different lifetimes and different owners:

- **Segment:** immutable device memory, such as weights. A broker owns it and processes map it read-only at a stable virtual address. It outlives any process.
- **Context:** the trust domain and the unit of time slicing. A fault stays in its context.
- **Process:** CPU state. It can run, park, hibernate, or not exist.

Mutable device memory, such as activations and KV caches, lasts for one request. A parked process holds none.

## What follows from the separation

| property | how it emerges | measured |
|---|---|---|
| native inference speed | clients use the normal driver and normal kernels | 11.74 ms mapped, 11.87 ms native, about 24 ms warmd |
| identical output | same weights at 4 KiB alignment | difference 0 on 7 architectures |
| deduplication | one physical copy, many mappings | 16 processes in 16.3 GB, against about 120 GB of copies (computed) |
| fault isolation | one context per process | writer faults; others keep the same output |
| write protection | read-only mapping | a kernel store into weights faults the writer |
| pause without losing state | unmap only | park 0.38 ms, resume 1.3 ms |
| no warm-up after eviction | kernels stay loaded in the process | first forward 13.4 ms after reload |
| isolation at no latency cost | swap is a map and one large copy | p99 219 ms, against 245 ms for one shared process |

The cold start is no longer one number. It is a ladder:

| state of the process | device memory while idle | time to first result |
|---|---|---:|
| mapped | segment + about 530 MiB | one forward pass |
| parked, segment resident | about 530 MiB | 1.3 ms + forward |
| hibernated, segment resident | 0 | about 241 ms; slower under load |
| evicted segment | none for the segment | add a copy: 118 ms mean for 1.6 GB |
| no process, zygote available, segment resident | 0 | 0.65 s, 7.5 GB model |
| full native start, page cache warm | 0 | 4.35 s, 7.5 GB model |

On a real engine, vLLM with Qwen2.5-0.5B, the lowest state holds 503 MiB of shared broker context and reaches the same output in 1.50 s. A real cold start takes 21.97 s (Result 20).

On ten real minutes of arrivals, isolated processes on segments reach a p99 of 141 ms, against 147 ms for one unisolated process and 3891 ms for a favoured scale-to-zero simulation (Result 19).

On sixteen vLLM processes of Qwen2.5-1.5B on one A100, with at most six active and KV pools included, park and hibernate serve 1.45 times the tokens of stopping and cold-starting engines. The p99 latency falls from 424 s to 132 s on an overloaded trace. Killing the broker mid-run caused no errors. With less device memory than plain (24.1 GiB against 25.6 GiB), and with park keeping the host half of the KV pool, the p50 falls from 148 s to 15 s and the p99 from 429 s to 103 s. A wake is 0.78 s: a cuda-checkpoint restore of 0.56 s and a segment reload of 0.23 s. A parked process holds 559 MiB, close to a bare context, so the restore is near its floor (Results 33 to 36).

vLLM's own sleep mode gets the same latency on the same trace: p50 11.9 s against 12.3 s. Its wake is 0.16 s, because a sleeping process keeps its context. It uses 5.3 GiB more device memory, 18% of the total (Result 37). So most of the gain over plain comes from one idea: keep the process alive, and free only its device state. That idea does not need segments. Segments add three things that sleep mode lacks:
- one read-only copy of the weights, shared by many processes or sandboxes;
- low device memory through hibernation;
- independence from the engine.

The sharing is the part that this trace does not test.

Result 38 tests the sharing. Each replica of one model is its own vLLM process and its own context, so the replicas are isolated. With one shared segment, a replica costs 0.94 GiB instead of 3.91 GiB, and 34 replicas fit at idle against 9. Seven replicas serve a trace at the same throughput with 10.6 GiB instead of 28.9 GiB. A replica that writes into the shared weights faults alone, and the others keep their output. On this compute-bound trace, the freed memory does not make the model faster. It makes room for more tenants. One unisolated process still has 2.4 times lower latency than seven isolated replicas, at the same throughput.

Segments cross trust boundaries. Two containers, another uid, a user namespace, no capabilities, and an exited owner all work with the unmodified driver. A gVisor sandbox works with a 9-line nvproxy change, and real kernels there give the same output with fault isolation (Results 23 to 25).

## Where it sits against warmd

warmd joined two ideas. The separation shows that they are independent choices:

| need | choice | measured |
|---|---|---|
| tenants that do not trust each other | one context each, shared segments | 3.76 new processes per second, isolated |
| one trust domain, new process per task | one shared context, shared segments | 78.87 tasks per second, 7.5 GB model |
| one trust domain, many short kernels at once | MPS, no shim | 74803 ops/s, against 60385 for warmd and 11667 for native contexts, RTX 3060 |

On the second row, a segment makes warmd 39 times faster at equal concurrency. MPS takes the third row from warmd. warmd keeps one unique property: a process can fork after CUDA init.

Two settings change the context cost more than any design here: persistence mode (cuInit 562 ms to 36 ms) and `CUDA_DEVICE_MAX_CONNECTIONS=1` (14.16 starts per second, against 4.81; restore 121 ms, against 229). The second holds on an A100 too: 12.37 starts per second against 4.36, and 10.69 restores per second against 5.86. It costs 17% when several streams could overlap, so it suits single-stream inference processes (Results 27 and 31).

## What it does not solve

- The start rate is about 12 to 14 contexts per second with one hardware queue, and restores about 11 to 12, on an RTX 3060 and an A100 (Results 27, 29 and 31).
- A context's device memory has no setting that reduces it: a bare context is 109 MiB on the RTX 3060 and 427 MiB on the A100, and PyTorch adds more. Only hibernation frees it.
- Host memory becomes the limit. A hibernated process costs 1412 MiB PSS, or an estimated 1137 MiB after a zygote fork (computed: 669 + 469). 62 GB holds about 50.
- Weights must be immutable. Training and in-place updates need ordinary memory.
- A segment frees weights only. vLLM keeps a 3.46 GiB KV pool unless the process releases it at park. Result 20 does this through engine internals, and the pool's contents are lost.
- A client must cooperate to release a mapping. The safe forced path is to kill it.
- The first load with a pinned copy is slow: 7.6 s for 7.5 GB. A pinned staging pool may help, at about 15 GB/s (Result 4); this is not integrated.
- Weights are not encrypted or hidden from tenants that map them. Share a segment only when both tenants may read it.

## What is not proven

- The real arrival trace comes from CPU functions (Result 18). It shows a large idle share for that shape, but no GPU fleet data was available. The real saving is unknown.
- One real checkpoint and vLLM 0.6.6 work (Result 16), on one small model. Other engines and larger models are not tested.
- Two GPU models, two driver branches, PyTorch only, no multi-GPU models.
- Hibernation, park and evict under load, and vLLM were not run in gVisor.
- The gVisor change has a first review (`NVPROXY_REVIEW.md`). It found a type confusion that version 1 exposed, and version 2 closes it with a device check. Version 2 passes on a GPU and rejects every host fd tried (Result 32). No maintainer has reviewed it.
- The fleet saving is a model: usable device memory divided by model size, so 2.4 to 24 times for 16 GB to 1.6 GB models, and none for a model that fills the GPU or when compute is the limit (Results 21 and 26). Packing creates this saving. Segments make it safe between tenants who do not trust each other.

## Prior work

The exact combination did not appear in two searches, which is weak evidence. Close work: SwapServeLLM (hot swap through GPU checkpoints), Prism (memory ballooning for many LLMs), Snowflake model hotswapping, LLM-Mesh, WAGES. They swap or share whole workloads. The difference here is that the immutable part leaves the process, so the snapshot and the swap no longer contain it.

## Next experiments, in order

1. Test the part that sleep mode cannot do: many tenants of one model in separate processes or sandboxes, sharing one copy of the weights. Measure how many active replicas fit on one GPU with segments and with sleep mode, and the cost of a fault in one of them. Also test sleep mode with cuda-checkpoint, which may close the memory gap. A zygote fork does not beat the 0.56 s restore for vLLM, because engine set-up alone takes 6.8 s. Done so far: the comparison with vLLM sleep mode (Result 37), the many-model run (Result 33), the wake split and an equal-memory rerun (Result 34), the KV fix (Result 35), the parked footprint (Result 36), and broker restart with reconnect.
2. Compare with one vLLM process that serves many LoRA adapters on the same trace. This applies only when the models share a base.
3. Add per-tenant memory accounting, and expose segments to the scheduler.
4. Support models sharded over several GPUs.
5. The nvproxy change is offered to gVisor as pull request 16898.

An independent review on 2026-11-03 named these gaps. It also found three code problems, now fixed: a reload while clients still mapped the old copy, a stale host copy after the file changed, and one lock for all loads.

## Reproduce

Scripts are in `scripts/`, run on the machine with the GPU, from `/workspace/exp`. Start with `make_models.py`, then the `run_*.sh` script for each result. The tools expect the warmd build at `/workspace/warmd`. The cross-container scripts (`run_h19*.sh`, `run_h21*.sh`, `run_h22.sh`, `run_h23.sh`) need a VM with Docker and the NVIDIA runtime.
