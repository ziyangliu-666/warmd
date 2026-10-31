# Errors

When the shim cannot connect, it prints the reason to stderr, for example
`[warmd] host refused session: all sessions in use`.

| code | where | cause | what to do |
|---|---|---|---|
| 3, `NOT_INITIALIZED` | `cuInit` | no daemon at `WARMD_SOCKET` within the connect timeout, or no active worker | start `warmd`; check `warmd_workers{state="active"}` |
| 2, `OUT_OF_MEMORY` | `cuInit` | all sessions in use and the wait timed out | raise `sessions` or `session-wait-ms` |
| 2, `OUT_OF_MEMORY` | allocation | the session quota or the GPU is full | raise `quota-mb` or free memory |
| 35, `INSUFFICIENT_DRIVER` | `cuInit` | the shim and the daemon use different protocol versions | use builds from the same release |
| 800, `NOT_PERMITTED` | `cuInit` | your uid is not allowed | add it with `allow-uid` |
| 46, `DEVICES_UNAVAILABLE` | any GPU call | the shared context was lost (a fault in some session, or a daemon restart) | run the task again in a new process; `ZygotePool` does this |
| 700, 714 to 719 | any GPU call | a device fault reached this session | the task may be the cause; run it alone to find out |
| 801, `NOT_SUPPORTED` | the call | the API is not implemented | see [compatibility.md](compatibility.md) |
| 1, `INVALID_VALUE` | free or destroy | the memory or handle belongs to another session or was already freed | fix the program |
