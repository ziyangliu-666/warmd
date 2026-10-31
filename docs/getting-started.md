# Getting started

This page builds warmd, starts a daemon, and runs a CUDA program and a PyTorch task pool through
it. You need a Linux machine with an NVIDIA GPU, driver 550 or newer, the CUDA toolkit headers,
`gcc`, `make`, and for the Python parts PyTorch built for CUDA 12.

## Build

```console
$ git clone https://github.com/ziyangliu-666/warmd && cd warmd
$ make
```

## Start the daemon

```console
$ build/warmd --metrics-port 9400
... INFO  [warmd 4100] listening on /tmp/warmd/gpu0.sock: gpu 0, 64 sessions x 16.0 MB, 1 standby, ...
... INFO  [warmd 4100] worker 4101 ready in 1104 ms (GPU mem used 862 MB)
... INFO  [warmd 4100] worker 4101 promoted to ACTIVE (epoch 1)
```

The daemon starts two worker processes. One owns the CUDA context and serves clients. The
other builds its own context and waits; it takes over if the first one dies.

## Run a CUDA program

`build/demo` is a small driver API program linked against the shim:

```console
$ build/demo
{"demo":1,"bad":0,"wrong":0,"sms":108,"init_ms":1.95,...,"steady_launch_sync_us":37.0,"total_ms":2.96}
```

For any other program, put the shim's directory first on the library path. Nothing else
changes:

```console
$ export LD_LIBRARY_PATH=$PWD/build/lib
$ python3 -c "import torch; print(torch.randn(4, 4, device='cuda').sum())"
```

The process connects to `/tmp/warmd/gpu0.sock` (set `WARMD_SOCKET` for another path) and gets
its own session. When the process exits, the daemon frees everything the session allocated.

## Run many short tasks

Install the Python package:

```console
$ pip install ./launcher
```

Task functions must be importable, so put them in a module:

```python
# tasks.py
import torch

def score(seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(256, 256, generator=g).cuda()
    return float((x @ x).softmax(1).sum())
```

```python
# run.py
from warmd import ZygotePool
import tasks

with ZygotePool(size=4) as pool:
    print(pool.map(tasks.score, range(1000))[:3])
```

```console
$ PYTHONPATH=. python3 run.py
```

Each zygote imports PyTorch and warms up CUDA and cuBLAS once. Each task then runs in a new
process forked from a zygote. See [pool.md](pool.md) for options and error handling.

## Look at the metrics

```console
$ curl -s localhost:9400/metrics | grep -E 'sessions_active|rpc_total|failovers'
warmd_sessions_active 0
warmd_failovers_total 0
warmd_rpc_total 3399
```

## Run the tests

```console
$ tests/run.sh
```

Next: [deploy.md](deploy.md) to run warmd as a service.
