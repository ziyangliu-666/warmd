# warmd

warmd keeps one warm NVIDIA® CUDA® context per GPU and lets short GPU tasks run in their own
processes without CUDA start-up.

A task process loads a replacement `libcuda.so.1` (the shim). The shim sends driver API calls
to the `warmd` daemon over a private shared-memory session. The task never creates a context,
never loads GPU code itself, and does not need access to `/dev/nvidia*`. Unmodified programs
work, including PyTorch with cuBLAS, cuDNN, Triton, NVRTC and CUDA graphs.

Because no CUDA state lives in the task process, a process that has already imported PyTorch
and warmed up cuBLAS can `fork()` and the child can use the GPU. With native CUDA a context
cannot cross `fork()`. The Python package `warmd` uses this to run each task as a fork of a
warm process.

## Why

Creating a CUDA context costs about 160 ms on an A100, and the driver does it one at a time:
we measured roughly five context lifecycles per second per GPU, however many CPUs try at once.
A system that starts a fresh process (and so a fresh context) for each short task stays at
that rate. warmd builds the context once and keeps it.

## Performance

A100-SXM4-40GB, driver 570.211.01, PyTorch 2.5.1, 8 vCPUs. Each task is a separate process that
runs a 256×256 matmul, a softmax and a few copies on its own random inputs; every result is
checked.

| how each task starts | tasks/s, one at a time | tasks/s, 8 at a time |
|---|---:|---:|
| native, new `python` process | 0.5 | 2.3 |
| native, fork after `import torch` | 2.1 | 4.5 |
| warmd, fork after `import torch` | 8.1 | 59 |
| warmd, fork after CUDA and cuBLAS warm-up | 42 | 133 |
| warmd, `ZygotePool` with 2 to 8 zygotes | | 200 to 230 |

The last rows are limited by CPU (fork plus Python work), not by the GPU. Lower-level numbers:
`cuInit` takes 2 ms instead of 162 ms; an empty kernel launch plus synchronize takes 30 µs
instead of 9 µs; large host/device copies run at 3 to 5 GB/s instead of 6 to 25 GB/s. See
[docs/performance.md](docs/performance.md) and the raw data in [results/](results/).

## What warmd is not

- It is not a security boundary between tenants. All sessions of one daemon share one CUDA
  context, and a kernel can read any memory in its context. Run one daemon per trust domain.
  Details: [docs/trust-model.md](docs/trust-model.md).
- It is not a scheduler or a cluster manager. It serves one GPU on one machine.
- It is not a full CUDA implementation. Managed memory, IPC, peer access, graphics interop and
  the explicit graph-building API are not supported. See
  [docs/compatibility.md](docs/compatibility.md).

## Building

Linux x86-64, an NVIDIA driver 550 or newer (570.x is the validated branch), the CUDA toolkit
headers, `gcc` and `make`.

```console
$ make
$ ls build
demo  driver_features  driver_features_native  lib  warmd
```

`build/warmd` is the daemon and `build/lib/libcuda.so.1` is the shim.

## Using

Start one daemon per GPU:

```console
$ build/warmd --gpu 0 --metrics-port 9400
... INFO  [warmd 4100] listening on /tmp/warmd/gpu0.sock: gpu 0, 64 sessions x 16.0 MB, 1 standby ...
... INFO  [warmd 4100] worker 4102 promoted to ACTIVE (epoch 2)
```

Run programs with the shim first on the library path:

```console
$ export LD_LIBRARY_PATH=$PWD/build/lib
$ python3 -c "import torch; x = torch.randn(512, 512, device='cuda'); print((x @ x).sum().item())"
```

Run many short tasks from warm processes:

```python
from warmd import ZygotePool
import mytasks

with ZygotePool(size=4) as pool:
    results = pool.map(mytasks.score, inputs)
```

If a task causes a device fault, the shared context is lost for every session of that daemon.
The daemon switches to a warm standby worker, and the pool runs the affected tasks again one at
a time, so only the task that faulted returns an error.

Start with [docs/getting-started.md](docs/getting-started.md). For services, containers and
Kubernetes see [docs/deploy.md](docs/deploy.md).

## Testing

```console
$ tests/run.sh
```

This starts its own daemons and runs 16 system tests, including 15 PyTorch checks and 11 driver
API checks whose results are compared with native CUDA. It needs a GPU and PyTorch.

## Documentation

| page | content |
|---|---|
| [getting-started.md](docs/getting-started.md) | build, run, first tasks |
| [deploy.md](docs/deploy.md) | systemd, Docker, CDI, Kubernetes |
| [operations.md](docs/operations.md) | health, logs, failover, sizing |
| [pool.md](docs/pool.md) | the Python `ZygotePool` |
| [configuration.md](docs/configuration.md) | daemon options and client environment |
| [metrics.md](docs/metrics.md) | Prometheus metrics |
| [compatibility.md](docs/compatibility.md) | supported drivers and APIs |
| [errors.md](docs/errors.md) | error codes and what to do |
| [design.md](docs/design.md) | how it works and why |
| [trust-model.md](docs/trust-model.md) | what is isolated and what is not |
| [performance.md](docs/performance.md) | measurements and known costs |
| [hidden-tables.md](docs/hidden-tables.md) | undocumented driver tables the shim provides |
| [new-driver.md](docs/new-driver.md) | validating a new driver branch |
| [protocol.md](docs/protocol.md) | wire protocol |

## Status

Early. Tested on one machine type (A100, driver 570.211.01). The Kubernetes and CDI files have
not been tested on a cluster. Plans are in [ROADMAP.md](ROADMAP.md).

## License

Apache 2.0. The layout of the undocumented driver tables and the integrity-check hash come from
[ZLUDA](https://github.com/vosen/ZLUDA) (MIT or Apache 2.0); see [NOTICE](NOTICE).

CUDA and NVIDIA are trademarks of NVIDIA Corporation. warmd is not affiliated with NVIDIA.
