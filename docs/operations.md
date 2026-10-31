# Operating warmd

## Health checks

| metric | healthy | if not |
|---|---|---|
| `warmd_workers{state="active"}` | 1 | 0 means a worker is starting or workers keep crashing; read the log |
| `warmd_workers{state="standby"}` | equals `standbys` | a standby is still warming up (about 1 s) or failed to start |
| `warmd_sessions_waiting` | 0 most of the time | sessions are the bottleneck; raise `sessions` |
| `warmd_sessions_wait_timeouts_total` | flat | clients gave up waiting and got an error at `cuInit` |
| `warmd_failovers_total` | flat | each step is one lost context (a device fault or a dead worker) |

## Logs

The daemon logs one line per event to stderr in the form `time LEVEL [component pid] message`.
`log-level = 3` adds a line for each session open and close.

| message | meaning |
|---|---|
| `worker N ready in X ms` | a worker built its CUDA context |
| `promoted to ACTIVE` | that worker now serves sessions |
| `sticky CUDA error E in session S` | a device fault; the worker exits on purpose |
| `context lost: K open session(s) notified` | failover; those sessions now return error 46 |
| `workers crash-looping` | six worker deaths in 10 s; respawning is paused. Check `nvidia-smi` and `dmesg`. |

## What happens on a device fault

1. A kernel faults, for example with an illegal address. CUDA marks the context as unusable for
   every user of that context.
2. The worker reports which session saw the fault and exits.
3. The supervisor marks all open sessions as lost. The reporting session keeps the fault code;
   the others get error 46 (`CUDA_ERROR_DEVICES_UNAVAILABLE`). It promotes the standby worker,
   which takes microseconds.
4. New sessions go to the new worker right away. A replacement standby starts in the background.
5. Old sessions keep returning their error until the process exits, as with native CUDA.
   `ZygotePool` retries the affected tasks one at a time.

Detecting the fault is the GPU's job and takes 0.4 to 0.6 s on an A100. In our tests, a new
task succeeded 20 to 150 ms after the fault was reported.

## Restarts

Restarting the daemon ends all sessions; clients get error 46. The code cache is rebuilt as
tasks arrive. The first PyTorch task after a restart pays 50 to 100 ms of module loading.

## Performance notes

- One call into the daemon costs 10 to 15 µs. A small PyTorch task makes 50 to 150 calls.
- Large copies between host and device go through the session's staging memory and run at 3 to
  5 GB/s. Programs that copy a lot will notice; small compute tasks will not.
- With `ZygotePool`, throughput is limited by CPU (fork and Python work per task). Size the pool
  to the number of cores.
