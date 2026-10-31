# Configuration

## Daemon

`warmd [options]` or `warmd --config FILE`. A config file has `key = value` lines with the same
names as the options, without `--`. Lines starting with `#` are comments. Options given after
`--config` override the file. `deploy/gpu0.conf.example` lists all keys.

| option | default | meaning |
|---|---|---|
| `socket` | `/tmp/warmd/gpu0.sock` | unix socket path; missing directories are created |
| `gpu` | `0` | CUDA device ordinal (after `CUDA_VISIBLE_DEVICES`) |
| `sessions` | `64` | maximum concurrent client sessions, 1 to 256 |
| `rpc-mb` | `8` | per-session staging area; the largest single staged copy chunk |
| `ring-mb` | `8` | per-session ring for asynchronous copies; copies up to a quarter of it are asynchronous |
| `standbys` | `1` | warm standby workers kept for failover |
| `threads` | `4` | dispatch threads in the active worker |
| `quota-mb` | `0` | device memory limit per session; 0 means no limit |
| `session-wait-ms` | `30000` | how long a new client waits when all sessions are in use; 0 refuses at once |
| `allow-uid` | none | allow this uid; repeat for more. The daemon's uid and root are always allowed. |
| `allow-all` | off | allow every local uid |
| `socket-mode` | `0666` | permission bits of the socket file, in octal |
| `metrics-port` | `0` | serve metrics on `127.0.0.1:PORT/metrics`; 0 turns it off |
| `log-level` | `2` | 0 error, 1 warning, 2 info, 3 debug |

Memory: each session is a memfd of `rpc-mb + ring-mb + 4 KiB`. Every worker maps and pins all
of them once at start (the same pages, so they are not counted twice). Each worker, active or
standby, holds one CUDA context.

## Client (the shim)

| variable | default | meaning |
|---|---|---|
| `LD_LIBRARY_PATH` | | must list the shim's directory first, so `libcuda.so.1` resolves to it |
| `WARMD_SOCKET` | `/tmp/warmd/gpu0.sock` | which daemon, and so which GPU, to use |
| `WARMD_CONNECT_TIMEOUT_MS` | `5000` | how long to wait for the socket to appear |
| `WARMD_SPIN_US` | `30` | busy-wait before sleeping on a reply; lower uses less CPU and adds latency |
| `WARMD_TRACE` | | `1` logs errors, unknown symbols and driver tables; `2` also logs every call |
| `WARMD_STATS` | | print per-process call statistics at exit |
| `WARMD_HASH_ALL` | | key every GPU code image by its content hash (default: by file identity when it comes from a mapped file) |
| `WARMD_UNK_NOTFOUND` | | report unimplemented symbols as not found instead of returning logging stubs |
| `WARMD_ALLOW_UNVALIDATED_DRIVER` | | silence the warning for driver branches that are not validated |

The shim exports two functions that are not part of CUDA:

- `int cuWarmdStatus(void)` returns 46 if this process lost its context, otherwise the first
  sticky fault code it saw (such as 700), otherwise 0.
- `void cuWarmdPrintStats(void)` prints the call statistics now.
