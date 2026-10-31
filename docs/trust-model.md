# Trust model

One warmd daemon is one trust domain. This page lists what the daemon separates between
sessions and what it cannot separate.

## Separated

| risk | how warmd handles it | test |
|---|---|---|
| a local user who should not use the GPU | socket mode plus a uid allowlist checked with `SO_PEERCRED` | `uid_allowlist` |
| a client reading another session's host-side data (staging, copies, kernel arguments) | each session is its own memfd and a client only receives its own | `isolation` |
| data left in a region for the next client | the supervisor zeroes every resident page before reuse | |
| freeing or destroying another session's memory, streams, events or graphs | the worker records owners and rejects foreign handles | `isolation` |
| one client using all device memory | `quota-mb` per session | `memory_quota` |
| a killed client leaking GPU memory | closing the socket frees everything (2 GB in about 20 ms) | `reclaim_on_kill` |
| a device fault blocking later work | failover to a warm standby | `fault_failover` |
| mismatched client and daemon versions | versioned handshake | `protocol_version_check` |

## Not separated

All sessions of a daemon share one CUDA context. CUDA has no memory protection between kernels
in the same context, so a kernel from session A can read and write session B's device memory.
NVIDIA MPS has the same property unless it uses static partitions.

Put tenants that do not trust each other on different daemons, which means different GPUs or
different MIG instances.

A device fault, whether by accident or on purpose, ends the context for every open session of
that daemon. Those sessions get error 46 and can retry. A client that faults on purpose and
repeatedly causes repeated failovers; block it before it reaches warmd.

## Other properties

- Task processes do not need access to `/dev/nvidia*`, so task code cannot reach the GPU
  kernel driver directly. We tested this with the device nodes made unreadable for the task's
  user: PyTorch still ran through warmd.
- The metrics port listens on 127.0.0.1 only.
- The daemon needs the GPU device nodes and enough `RLIMIT_MEMLOCK` to pin session memory. The
  systemd unit sets the limit.

To report a vulnerability, see [SECURITY.md](../SECURITY.md).
