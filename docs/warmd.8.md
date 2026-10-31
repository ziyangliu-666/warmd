# warmd 8

## NAME

warmd - hold a CUDA context for one GPU and serve driver calls from other processes

## SYNOPSIS

`warmd [--config FILE] [OPTION VALUE]...`

## DESCRIPTION

warmd creates a CUDA context on one GPU and serves the CUDA driver API to client processes that
load the warmd `libcuda.so.1`. Each client gets a session: a private memory region and an
eventfd, passed over a unix socket. When the client's socket closes, warmd frees everything the
session allocated.

The process started by the user is the supervisor. It never initializes CUDA. It starts worker
processes (the same binary with `--worker`): one active worker owns the context and serves
sessions, the others build a context and wait. If the active worker exits, for example after a
sticky device fault, every open session is marked lost and a waiting worker takes over.

## OPTIONS

Options can also be given in the file named by `--config`, one `name = value` per line, without
the leading dashes. Options after `--config` override the file.

`--socket PATH`
: Socket path. Default `/tmp/warmd/gpu0.sock`.

`--gpu N`
: CUDA device ordinal. Default 0.

`--sessions N`
: Maximum concurrent sessions, 1 to 256. Default 64.

`--rpc-mb N`, `--ring-mb N`
: Size of each session's staging area and asynchronous copy ring. Default 8 each. Every worker
  pins all session memory at start: 64 sessions of 16 MB pin 1 GB.

`--standbys N`
: Workers kept waiting for failover. Default 1. Each holds its own context.

`--threads N`
: Dispatch threads in the active worker. Default 4.

`--quota-mb N`
: Device memory limit per session. 0 (default) means none. Clients see the quota as the device's
  total memory.

`--session-wait-ms N`
: How long a client waits for a free session before it is refused. Default 30000.

`--allow-uid UID`, `--allow-all`
: Uids allowed to connect, besides root and the daemon's own uid.

`--socket-mode MODE`
: Socket file mode, octal. Default 0666.

`--metrics-port PORT`
: Serve Prometheus metrics on 127.0.0.1:PORT. Default off.

`--log-level N`
: 0 errors, 1 warnings, 2 info (default), 3 debug.

## CLIENT ENVIRONMENT

`LD_LIBRARY_PATH`
: Must find the warmd `libcuda.so.1` before NVIDIA's.

`WARMD_SOCKET`
: Daemon socket. Default `/tmp/warmd/gpu0.sock`.

`WARMD_CONNECT_TIMEOUT_MS`
: How long to wait for the socket to appear. Default 5000.

`WARMD_SPIN_US`
: Busy-wait before sleeping on a reply. Default 30.

`WARMD_TRACE`
: 1 logs errors and unimplemented calls, 2 logs every call.

`WARMD_STATS`
: Print call counts and times at exit.

`WARMD_HASH_ALL`
: Identify cached GPU code by content hash only (default: by file and offset when possible).

`WARMD_ALLOW_UNVALIDATED_DRIVER`
: Do not warn on driver branches other than 570.

The client library also exports `int cuWarmdStatus(void)`: 46 if the session was lost, else the
first sticky fault code seen (such as 700), else 0.

## ERRORS

Errors specific to warmd, as seen by CUDA callers:

| code | meaning |
|---|---|
| 3 | no daemon or no active worker at `cuInit` |
| 2 | all sessions busy and the wait timed out (at `cuInit`), or quota exceeded |
| 35 | client and daemon protocol versions differ |
| 800 | uid not allowed |
| 46 | context lost (failover or daemon restart); retry in a new process |
| 801 | call not implemented, see compat.md |

## METRICS

`warmd_sessions_active`, `warmd_sessions_capacity`, `warmd_sessions_waiting`,
`warmd_sessions_total`, `warmd_sessions_waited_total`, `warmd_sessions_wait_timeouts_total`,
`warmd_sessions_rejected_total{reason}`, `warmd_workers{state}`,
`warmd_worker_restarts_total`, `warmd_failovers_total`, `warmd_faults_total`,
`warmd_rpc_total`, `warmd_rpc_seconds_total`, `warmd_rpc_op_total{op}`,
`warmd_kernel_launches_total`, `warmd_copy_bytes_total{dir}`,
`warmd_code_cache_hits_total`, `warmd_code_cache_misses_total`,
`warmd_session_device_memory_bytes`, `warmd_gpu_memory_total_bytes`,
`warmd_gpu_memory_free_bytes`.

Counters include workers that have exited. Operation numbers are defined in `src/proto.h`.

## SIGNALS

SIGTERM and SIGINT stop the workers and remove the socket.

## FILES

`contrib/gpu0.conf.example`, `contrib/systemd/warmd@.service`.

## SEE ALSO

[deploy.md](deploy.md), [design.md](design.md), [compat.md](compat.md)
