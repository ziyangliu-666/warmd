# Segments

A segment is read-only device memory that outlives the processes that use it. A broker loads a safetensors file once, and each client maps it without a copy.

A client keeps its own CUDA context and uses NVIDIA's driver library, not warmd's. A fault stays in the client that caused it, and a store into a segment faults the writer.

## Running

```console
$ make
$ python3 -m warmd.segments --socket /run/warmd/seg.sock --file llama=/models/llama.safetensors
```

Only registered names can be opened. Clients cannot name a path.

```python
from warmd.segments import Client

c = Client("/run/warmd/seg.sock")
weights = c.open("llama")             # {tensor name: read-only CUDA tensor}
model.load_state_dict(weights, assign=True)
```

## Park and resume

`c.park("llama")` unmaps the segment and frees the process's scratch memory. The tensors keep their addresses. `c.resume("llama")` maps the segment again at the same addresses, so the model needs no rebuild and no warm-up.

If the broker evicted the segment, `resume` waits while the broker loads it again, and returns True. If the file changed, the tensors then hold the new weights. If its layout changed, `resume` fails.

| step, 1.6 GB model, A100 | time |
|---|---:|
| park | 0.4 ms |
| resume, segment loaded | 1.3 ms |
| resume after eviction | about 120 ms |

To free the context's device memory too (about 500 MiB for a PyTorch process on an A100), checkpoint the parked process with `cuda-checkpoint --toggle`. Restore took 121 ms on an RTX 3060 with `CUDA_DEVICE_MAX_CONNECTIONS=1`, and 229 ms without it.

## Memory budget

`--budget-gib 30` caps the device memory of all segments. To load a new one past the cap, the broker evicts the least recently used segments that no client maps. If every loaded segment is mapped, the open fails.

## Broker restart

Mappings survive a broker crash: clients keep running on the memory they map. A client reconnects on its next call. After a restart, `resume` loads a fresh copy and returns True; the old copy is freed when the last client parks.

## Access

- The socket is mode 0660 by default. `--allow-uid` adds uids that may open segments.
- Only the broker's uid and root may evict.
- An fd received from the broker grants read access to the segment. Do not pass it on.
- A segment works across containers. A gVisor sandbox needs a change in nvproxy: see `research/gvisor_issue_draft.md`.

## Limits

- Weights must not change. Training needs ordinary memory.
- The broker refuses to evict a segment while any client maps it, because a later load would hold a second copy. `evict(force=True)` overrides this. A client that exits releases its mappings.
- The first load copies the file into pinned host memory, which stays allocated for later reloads.

Measurements and the experiments behind them are in `research/`.
