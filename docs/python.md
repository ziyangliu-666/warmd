# Python package

```console
$ pip install ./python
```

`warmd.ZygotePool` starts a few zygote processes. Each imports PyTorch and runs a warm-up
function once, then forks one child per task. Tasks run in separate processes; arguments and
results are pickled.

```python
from warmd import ZygotePool
import mytasks          # task functions must be importable at module level

with ZygotePool(size=4) as pool:
    results = pool.map(mytasks.score, range(1000))
```

`ZygotePool` arguments:

| argument | default | |
|---|---|---|
| `size` | 2 | zygote processes; scale with CPU cores |
| `preload` | `("torch",)` | modules imported by each zygote |
| `warmup` | `"warmd.warm:torch_cuda"` | function run once per zygote |
| `max_inflight` | 8 | tasks running at once per zygote |
| `retry` | True | rerun tasks hit by a device fault |
| `env` | `os.environ` | environment for the zygotes, including `PYTHONPATH` |

Without warmd, use `warmup="warmd.warm:torch_import_only"`: a native CUDA context must not exist
before `fork()`.

Do not use CUDA tensors created during warm-up inside tasks. They belong to the zygote's session.
Keep shared data on the CPU and copy it in the task.

## Device faults

A fault kills the shared context, so every task running at that moment fails. With `retry=True`
the pool waits for running tasks to finish, restarts its zygotes, and runs the failed tasks one
at a time. A task that faults again on its own raises `TaskError` with `kind == "fault"`. The
others return normally.

Other `TaskError.kind` values: `"error"` (the task raised), `"crashed"` (the task process was
killed), `"lost"` (the context was lost again on every retry). `TaskError.remote_traceback`
holds the task's traceback.
