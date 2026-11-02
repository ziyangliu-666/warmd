# Separate lifetimes: segments, contexts and processes

Status: a research result from one A100, driver 570, on 2026-11-02 and 2026-11-03. See `LOG.md` for data and failures. It is not a production design.

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

## Where it sits against warmd

warmd joined two ideas. The separation shows that they are independent choices:

| need | choice | measured |
|---|---|---|
| tenants that do not trust each other | one context each, shared segments | 3.76 new processes per second, isolated |
| one trust domain, new process per task | one shared context, shared segments | 78.87 tasks per second, 7.5 GB model |
| one trust domain, many short kernels at once | shared context | 22980 ops/s, against 16694 with a fair lease |

warmd stays useful for the second and third rows. On the second row, a segment makes warmd 39 times faster at equal concurrency.

## What it does not solve

- The driver limit stays. New contexts and restores both run at about 3 to 6 per second per GPU.
- Host memory becomes the limit. A hibernated process costs 1412 MiB PSS, or an estimated 1137 MiB after a zygote fork (computed: 669 + 469). 62 GB holds about 50.
- Weights must be immutable. Training and in-place updates need ordinary memory.
- A segment frees weights only. vLLM keeps a 3.46 GiB KV pool unless the process releases it at park. Result 20 does this through engine internals, and the pool's contents are lost.
- A client must cooperate to release a mapping. The safe forced path is to kill it.
- The first load with a pinned copy is slow: 7.6 s for 7.5 GB. A pinned staging pool may help, at about 15 GB/s (Result 4); this is not integrated.
- Weights are not encrypted or hidden from tenants that map them. Share a segment only when both tenants may read it.

## What is not proven

- The real arrival trace comes from CPU functions (Result 18). It shows a large idle share for that shape, but no GPU fleet data was available. The real saving is unknown.
- One real checkpoint and vLLM 0.6.6 work (Result 16), on one small model. Other engines and larger models are not tested.
- One GPU model, one driver branch, PyTorch only, no multi-GPU models.
- A second uid works: uid 65534 mapped a root broker's segment, with the same output and 12.08 ms per forward pass. Everything ran in one container, because the rented container cannot create namespaces. A forum report says exported fds do not cross containers, and NVIDIA's documentation ties multi-user isolation to IMEX channels. Neither is tested. This is the largest open risk.
- Device-memory files under gVisor are untested.

## Prior work

The exact combination did not appear in two searches, which is weak evidence. Close work: SwapServeLLM (hot swap through GPU checkpoints), Prism (memory ballooning for many LLMs), Snowflake model hotswapping, LLM-Mesh, WAGES. They swap or share whole workloads. The difference here is that the immutable part leaves the process, so the snapshot and the swap no longer contain it.

## Next experiments, in order

1. Share a segment across two containers. This needs a machine that allows namespaces. If it fails, the multi-tenant case needs another path.
2. Measure the idle share on a GPU fleet trace. If it is low, there is little to reclaim.
3. Turn the vLLM result into a supported interface, and test a larger model with requests in flight.
4. Run under gVisor with a GPU.

## Reproduce

Scripts are in `scripts/`, run on the machine with the GPU, from `/workspace/exp`. Start with `make_models.py`, then the `run_*.sh` script for each result. The tools expect the warmd build at `/workspace/warmd`.
