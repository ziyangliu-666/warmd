# Roadmap

Ordered by measured impact (see docs/performance.md).

1. Return immediately from calls that only enqueue work (launch, memset, async copy, event
   record) and report their errors at the next synchronization. Goal: launch rate close to
   native.
2. Share pinned host memory with the daemon so pinned copies go straight to the GPU, and
   double-buffer staging for pageable copies.
3. Give multi-threaded clients more than one request slot so their threads do not wait on each
   other.
4. A Kubernetes device plugin that advertises one resource per session; test the DaemonSet and
   CDI files on a cluster.
5. Validate more driver branches (550, 560, 575 and later) on self-hosted GPU CI runners.
6. A client-side GPU selector for machines with several daemons.
7. Optional hardware partitioning per daemon (MIG or green contexts).
8. More of the driver API (explicit graph building, virtual memory management, memory pools)
   when real programs need it. `WARMD_TRACE=1` shows what a program calls.
