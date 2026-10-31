# Metrics

With `metrics-port` set, the daemon serves Prometheus text format at
`http://127.0.0.1:PORT/metrics`. Counters keep counting across failovers: when a worker dies,
the supervisor adds its counters to a running total.

| metric | type | meaning |
|---|---|---|
| `warmd_sessions_active` | gauge | open sessions |
| `warmd_sessions_capacity` | gauge | the `sessions` setting |
| `warmd_sessions_waiting` | gauge | clients waiting for a free session |
| `warmd_sessions_total` | counter | sessions opened |
| `warmd_sessions_waited_total` | counter | clients that had to wait |
| `warmd_sessions_wait_timeouts_total` | counter | clients that waited too long |
| `warmd_sessions_rejected_total{reason}` | counter | refused connections: `version`, `perm`, `full`, `noworker`, `proto` |
| `warmd_workers{state}` | gauge | workers by state: `active`, `standby`, `starting` |
| `warmd_worker_restarts_total` | counter | worker processes that died |
| `warmd_failovers_total` | counter | times the active worker was lost |
| `warmd_faults_total` | counter | sticky device faults reported by workers |
| `warmd_rpc_total` | counter | calls served |
| `warmd_rpc_seconds_total` | counter | time spent serving calls |
| `warmd_rpc_op_total{op}` | counter | calls per operation; numbers are defined in `proto/warmd_proto.h` |
| `warmd_kernel_launches_total` | counter | kernel launches |
| `warmd_copy_bytes_total{dir}` | counter | bytes copied, `h2d` or `d2h` |
| `warmd_code_cache_hits_total` | counter | GPU code images found in the shared cache |
| `warmd_code_cache_misses_total` | counter | GPU code images loaded |
| `warmd_session_device_memory_bytes` | gauge | device memory currently allocated by sessions |
| `warmd_gpu_memory_total_bytes` | gauge | device memory, last sample |
| `warmd_gpu_memory_free_bytes` | gauge | free device memory, last sample |
