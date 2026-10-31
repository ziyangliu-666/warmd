# ZygotePool

`warmd.ZygotePool` runs each task in a new process forked from a warm "zygote" process. A
zygote imports your modules and runs a warm-up function once; every fork inherits that state.

```python
from warmd import ZygotePool, TaskError

pool = ZygotePool(
    size=4,                              # zygote processes; scale with CPU cores
    preload=("torch",),                  # modules each zygote imports once
    warmup="warmd.warm:torch_cuda",      # called once per zygote before any fork
    max_inflight=8,                      # tasks running at once per zygote
    retry=True,                          # rerun tasks hit by a device fault, one at a time
)
fut = pool.submit(mymodule.task, 1, key="x")   # or pool.submit("mymodule:task", ...)
print(fut.result())
print(pool.map(mymodule.task, range(100)))
pool.close()
```

## Rules

- Task functions must be importable by the zygotes: define them at module level and pass a
  `PYTHONPATH` that contains the module (through `env=` or the environment).
- Arguments and return values are pickled. Return CPU values, not CUDA tensors.
- Each task runs in its own process, so tasks do not share Python state. Files you write are
  shared, as usual.
- A warm-up function can load a model. Forked tasks see the Python objects. Do not use CUDA
  tensors created during warm-up inside tasks: they belong to the zygote's session. Keep shared
  weights on the CPU in the zygote and copy them to the GPU in the task.
- Without warmd (native CUDA), use `warmup="warmd.warm:torch_import_only"`. A native CUDA
  context must not exist before `fork()`.

## Errors

A failed task raises `TaskError` from `future.result()`. `TaskError.kind` is one of:

| kind | meaning |
|---|---|
| `error` | the task raised an exception |
| `fault` | the task caused a device fault; it faulted again when it ran alone |
| `lost` | the GPU context was lost again during each retry |
| `crashed` | the task process died from a signal |

`TaskError.remote_traceback` has the task's traceback.

## How fault handling works

A device fault ends the shared context, so tasks running at the same time fail too, and the
pool cannot yet tell which task caused it. The pool:

1. marks every task that failed with a device error as a suspect;
2. waits until nothing is running;
3. restarts its zygotes, because their CUDA handles belong to the lost context;
4. runs the suspects one at a time, alone on the GPU.

A suspect that succeeds alone returns its result normally. A suspect that faults alone is the
cause and fails with `kind="fault"`. In our tests six tasks ran next to one faulting task: all
six returned correct results and only the faulting task failed.

The shim tells the pool what happened through `cuWarmdStatus()`, so the pool does not have to
parse error messages.
