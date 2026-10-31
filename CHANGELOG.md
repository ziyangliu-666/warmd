# Changelog

## 0.1.0 (unreleased)

First version.

- `warmd` daemon: a supervisor without CUDA, one active CUDA worker and warm standby workers
  for failover. Sessions are private memfd regions handed out over a unix socket with a
  versioned handshake and a uid allowlist. Clients wait in a queue when all sessions are busy.
  Per-session memory quotas, ownership checks, cleanup when a client exits, Prometheus metrics,
  config file, multi-threaded dispatch, shared GPU code cache.
- Shim (`libcuda.so.1`): the driver API used by PyTorch, cuBLAS and cuBLASLt, cuDNN, Triton,
  NVRTC and CUDA graphs captured from streams; async and 2D copies; `cuLink*`;
  `cuLaunchKernelEx`; the undocumented driver tables cudart and cuBLASLt need on driver 570;
  weak stubs for every other driver export; `fork()` support.
- Python package `warmd` with `ZygotePool`: warm fork servers that retry tasks hit by a device
  fault one at a time.
- 16 system tests, 15 PyTorch checks and 11 driver API checks; benchmarks; systemd, Docker, CDI
  and Kubernetes files (the last two not yet tested on a cluster).
