# warmd

warmd lets short CUDA programs skip CUDA context creation.

A daemon holds one CUDA context per GPU. Programs load the `libcuda.so.1` built here instead of
NVIDIA's and send their driver calls to the daemon over shared memory. They never open
`/dev/nvidia*`. PyTorch, cuBLAS, cuDNN, Triton and CUDA graphs run unmodified.

Since the program holds no CUDA state of its own, it can `fork()` after CUDA is initialized. The
`warmd` Python package starts each task as a fork of a process that has already imported
PyTorch and warmed up cuBLAS.

## Building

Needs Linux x86-64, NVIDIA driver 550 or newer, CUDA headers, gcc and make.

```console
$ make
$ build/warmd &
$ LD_LIBRARY_PATH=build/lib python3 -c 'import torch; print(torch.ones(2, device="cuda"))'
```

`tests/run.sh` runs the tests.

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

[warmd(8)](docs/warmd.8.md), [deploying](docs/deploy.md), [Python package](docs/python.md),
[design](docs/design.md), [compatibility](docs/compat.md), [performance](docs/perf.md).

## License

Apache 2.0. See [NOTICE](NOTICE).
