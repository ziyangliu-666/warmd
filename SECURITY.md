# Security

Read [docs/trust-model.md](docs/trust-model.md) first. One warmd daemon is one trust domain:
its sessions share one CUDA context and are not protected from each other's device memory.
That is documented behaviour, not a vulnerability.

Please report privately anything that lets a client read or change another session's host-side
data, use another session's handles, get past the uid allowlist or quotas, crash or hang the
supervisor, or keep resources after its session ends. Use GitHub's private vulnerability
reporting for this repository and include a way to reproduce the problem. We aim to reply
within seven days.
