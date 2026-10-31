// warmd_proto.h - wire protocol between the shim (libcuda.so.1 in task processes), the warmd
// supervisor, and its CUDA worker processes. Versioned; any layout change bumps WD_PROTO_VERSION.
//
// Connection: client connects to a SOCK_SEQPACKET unix socket, sends wd_hello, receives wd_welcome
// plus two fds (SCM_RIGHTS): [0] memfd of its *own* session region, [1] eventfd doorbell.
// A client never maps any other session's memory.
//
// Region layout: [wd_page (4 KiB header)] [rpc staging: rpc_size] [async ring: ring_size]
// Request: client fills op/a[], writes data to staging, sets state=WD_REQ, writes 1 to the eventfd.
// Reply:   worker fills r[]/err, sets state=WD_DONE, FUTEX_WAKEs `state`.
// Liveness: if the worker dies the supervisor sets `dead` (sticky error code) and wakes the client.
#ifndef WD_PROTO_H
#define WD_PROTO_H
#include <stdint.h>

#define WD_MAGIC          0x31444d57u   /* "WMD1" */
#define WD_PROTO_VERSION  2u
#define WD_PAGE           4096u
#define WD_DEF_RPC_MB     8u
#define WD_DEF_RING_MB    8u
#define WD_MAX_SESSIONS   256

enum { WD_IDLE = 0, WD_REQ = 1, WD_DONE = 2 };

// welcome.status
enum { WD_OK = 0, WD_E_VERSION = 1, WD_E_PERM = 2, WD_E_FULL = 3, WD_E_NOWORKER = 4, WD_E_PROTO = 5 };

// error returned to the CUDA caller when the device context was lost (worker died). Retryable.
#define WD_ERR_DEVICE_LOST 46   /* CUDA_ERROR_DEVICES_UNAVAILABLE */

typedef struct { uint32_t magic, version, pid, flags; } wd_hello_t;
typedef struct {
  uint32_t magic, version; int32_t status; uint32_t session_id;
  uint64_t region_size, rpc_off, rpc_size, ring_off, ring_size;
  uint64_t quota;                 // bytes of device memory this session may allocate (0 = unlimited)
  int32_t  driver_version, device_ordinal;
  uint32_t epoch, _p0;            // worker generation: changes on failover (handles from older epochs are invalid)
  char     reason[96];
} wd_welcome_t;

typedef struct {
  uint32_t magic, version;
  volatile int32_t state;         // futex word: WD_IDLE / WD_REQ / WD_DONE
  int32_t  err;
  int32_t  op, _p0;
  uint64_t a[16], r[16];
  volatile int32_t dead;          // != 0: context lost; every call returns this error
  uint32_t session_id, epoch;
  uint32_t _p1;
  uint64_t hiwater;               // worker: highest data byte it wrote (informational)
} wd_page_t;
_Static_assert(sizeof(wd_page_t) <= WD_PAGE, "page header too large");

#define WD_DATA(p) ((uint8_t*)(p) + WD_PAGE)

// RPC ops
enum {
  OP_NOP = 1, OP_DEV_INFO, OP_MEM_ALLOC, OP_MEM_FREE, OP_MEMINFO, OP_BLOB, OP_LIB_LOOKUP, OP_LIB_LOAD,
  OP_LIB_GET_MODULE, OP_LIB_GET_KERNEL, OP_KERNEL_GET_FUNC, OP_LIB_GET_GLOBAL, OP_MOD_GET_GLOBAL,
  OP_GET_FUNC, OP_PARAMINFO, OP_FUNC_GET_ATTR, OP_FUNC_SET_ATTR, OP_OCCUPANCY, OP_LAUNCH,
  OP_MEMSET, OP_H2D, OP_D2H, OP_D2D, OP_H2D_ASYNC, OP_D2H_ASYNC, OP_PENDING_QUERY, OP_RING_SYNC,
  OP_SYNC, OP_STREAM_QUERY, OP_STREAM_CREATE, OP_STREAM_DESTROY, OP_STREAM_WAIT_EVENT,
  OP_EVENT_CREATE, OP_EVENT_DESTROY, OP_EVENT_RECORD, OP_EVENT_SYNC, OP_EVENT_QUERY, OP_EVENT_ELAPSED,
  OP_CTX_SYNC, OP_CTX_LIMIT,
  OP_CAPTURE_BEGIN, OP_CAPTURE_END, OP_CAPTURE_INFO, OP_GRAPH_INSTANTIATE, OP_GRAPH_LAUNCH,
  OP_GRAPH_EXEC_DESTROY, OP_GRAPH_DESTROY,
  OP_LINK_CREATE, OP_LINK_ADD, OP_LINK_COMPLETE, OP_LINK_DESTROY, OP_MEMCPY2D, OP_GRAPH_NODES,
  OP_MAX
};

// supervisor <-> worker control messages (SOCK_SEQPACKET socketpair)
enum { C_ATTACH = 1, C_OPEN, C_CLOSE, C_PROMOTE, C_STOP,           /* supervisor -> worker */
       C_READY = 32, C_ACTIVE, C_CLOSED, C_FAULT };                /* worker -> supervisor */
typedef struct { uint32_t type, region; uint64_t arg[4]; } wd_ctl_t;

// stats page shared worker -> supervisor (for /metrics)
typedef struct {
  volatile uint64_t rpc_total, rpc_ns, op_count[OP_MAX], mem_used, lib_hits, lib_miss, launches,
                    bytes_h2d, bytes_d2h, faults, sessions_served;
  volatile uint64_t gpu_mem_total, gpu_mem_free;
} wd_stats_t;
#endif
