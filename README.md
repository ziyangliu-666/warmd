# warmd

Elastic GPU processes for PyTorch: share model weights between processes without a copy, and park or hibernate a process without losing its state.

**Segments** (`warmd.segments`) hold model weights in device memory that outlives any process. Each process maps them read-only and keeps its own CUDA context, so a fault stays in that process. A parked process resumes in about 1 ms. See [docs/segments.md](docs/segments.md).

**The warmd daemon** holds one CUDA context per GPU. Programs that load warmd's `libcuda.so.1` send their driver calls to it, so they can `fork()` after CUDA is initialized. Use it for many short tasks in one trust domain. All its processes share one fault domain.

![Without warmd](docs/img/without-warmd.svg)

![With warmd](docs/img/with-warmd.svg)

## Building

Needs Linux x86-64, NVIDIA driver 550 or newer, CUDA headers, gcc and make.

```console
$ make
$ build/warmd &
$ LD_LIBRARY_PATH=build/lib python3 -c 'import torch; print(torch.ones(2, device="cuda"))'
```

`tests/run.sh` runs the tests. `tests/segment_tests.py` needs no GPU.

## Performance

Short PyTorch tasks on an A100 (driver 570, 8 vCPUs), each in its own process:

| | 1 at a time | 8 at a time |
|---|---:|---:|
| native, fork after `import torch` | 2.1/s | 4.5/s |
| warmd, fork after `import torch` | 8.1/s | 59/s |
| warmd, fork after CUDA warm-up | 42/s | 133/s |

Each call to the daemon adds 10 to 15 µs, and large copies run at 3 to 5 GB/s. Details in
[docs/perf.md](docs/perf.md).

## Documentation

[segments](docs/segments.md), [warmd(8)](docs/warmd.8.md), [deploying](docs/deploy.md), [Python package](docs/python.md),
[design](docs/design.md), [compatibility](docs/compat.md), [performance](docs/perf.md).

## License

Apache 2.0. See [NOTICE](NOTICE).
