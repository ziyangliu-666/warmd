# Contributing

## Building and testing

```console
$ make
$ tests/run.sh      # needs an NVIDIA GPU and PyTorch
$ bench/run.sh      # optional; writes results/bench-<date>/
```

A change to the shim or the daemon must keep `tests/run.sh` passing, with PyTorch and driver
API results equal to native CUDA.

## Adding a driver API function

1. Implement it in `shim/shim.c`. The weak stub in `shim/stubs.c` is overridden automatically.
2. If it needs the GPU, add an operation to `proto/warmd_proto.h`. Increase `WD_PROTO_VERSION`
   if a structure layout changes.
3. Handle the operation in `host/worker.c`. Record any resource it creates on the session, so
   it is freed when the session ends, and reject handles that belong to other sessions.
4. Add a check to `tests/driver_features.c` or `tests/torch_conformance.py`.

## Other changes

- New driver branch: follow `docs/new-driver.md`.
- Documentation: follow `docs/STYLE.md`; `tools/style-check.sh` points out common problems.
- C code is C11. Avoid allocation on the per-call path. Python code needs 3.9 or newer and only
  the standard library.
- Keep one topic per commit. If a change affects performance, say what you measured and on
  which hardware.
