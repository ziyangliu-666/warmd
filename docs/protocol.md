# Protocol

The definitions are in [`proto/warmd_proto.h`](../proto/warmd_proto.h). This page describes
version 2. Any change to the layout increases `WD_PROTO_VERSION`, and the daemon refuses clients
with another version.

## Handshake

The socket is a unix `SOCK_SEQPACKET` socket.

1. The client sends `wd_hello_t`: magic `"WMD1"`, protocol version, pid, flags.
2. The daemon checks the magic, the version, the client's uid (`SO_PEERCRED`), that a worker is
   active, and that a session region is free. If none is free it holds the client in a queue.
3. The daemon replies with `wd_welcome_t` (status, session id, region size, staging and ring
   offsets and sizes, quota, worker epoch, reason text) and, on success, two file descriptors
   with `SCM_RIGHTS`: the session's memfd and its eventfd.
4. The connection stays open for the life of the session. When it closes for any reason, the
   worker frees the session's device memory, streams, events, graphs and link states, and the
   supervisor zeroes the region.

## Session memory

One memfd per session; a client maps only its own.

```
offset 0       wd_page_t (4 KiB): state futex, op, a[16] arguments, r[16] results, err, dead, ids
offset 4 KiB   staging area (rpc-mb): kernel parameters, names, copy chunks
then           async ring (ring-mb): async host-to-device sources, deferred device-to-host data
```

A call: the client writes `op`, `a[]` and any payload, sets `state = REQ` and writes to the
eventfd. A worker thread wakes (each eventfd is registered with `EPOLLONESHOT`), runs the call,
writes `r[]` and `err`, sets `state = DONE` and wakes the futex. The client spins for up to
`WARMD_SPIN_US`, then waits on the futex in 200 ms steps, checking `dead` and the socket in
between.

`dead` is set by the supervisor when the worker died; from then on every call returns that
value. `epoch` names the worker generation. A forked child whose handles come from a different
epoch fails at once instead of using invalid handles.

A device-to-host async copy into pinned memory is written to the ring with an event recorded
after it. At the next point where the program can see completion (stream, event or context
synchronize, or a successful query), the client asks which of those events finished and copies
that data to its destination.

## Supervisor and worker

They talk over a `SOCK_SEQPACKET` socketpair with `wd_ctl_t {type, region, arg[4]}` messages,
plus file descriptors where needed.

| direction | messages |
|---|---|
| supervisor to worker | `ATTACH` (region fds, stats page), `OPEN`, `CLOSE`, `PROMOTE`, `STOP` |
| worker to supervisor | `READY`, `ACTIVE`, `CLOSED`, `FAULT(region, code)` |
