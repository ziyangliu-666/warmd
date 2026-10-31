# Security

Report vulnerabilities through GitHub's private vulnerability reporting.

Sessions of one daemon share a CUDA context, so device memory is not isolated between them
([design.md](docs/design.md#isolation)).
