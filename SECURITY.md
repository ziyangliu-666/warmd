# Security

Sessions of one daemon share a CUDA context and can read each other's device memory. That is by
design (see docs/design.md) and not a vulnerability.

Report privately, through GitHub's private vulnerability reporting, anything that lets a client
read or change another session's host memory, use another session's handles, connect without
being allowed, exceed its quota, crash the supervisor, or keep resources after it exits.
