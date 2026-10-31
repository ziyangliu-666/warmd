// shim.c - drop-in libcuda.so.1 for task processes. Implements the CUDA *driver* API by forwarding
// calls to warmd over a private shared-memory session. The task process performs ZERO CUDA init:
// no context, no module load, no GPU memory of its own. Handles are opaque host-side values; device
// pointers are real (allocated in the host's context).
//
// Use:  LD_LIBRARY_PATH=<dir with this libcuda.so.1>  WARMD_SOCKET=/tmp/warmd/gpu0.sock  <program>
// Env:  WARMD_SOCKET   host socket (default /tmp/warmd/gpu0.sock)
//       WARMD_CONNECT_TIMEOUT_MS  wait for the host to come up (default 5000)
//       WARMD_TRACE=1|2  log resolves/errors (2: every RPC);  WARMD_STATS=1  per-process RPC stats
//       WARMD_SPIN_US  busy-wait before sleeping on a reply (default 30)
#define _GNU_SOURCE
#include "../proto/warmd_proto.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <elf.h>
#include <dlfcn.h>
#include <poll.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <linux/futex.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>

typedef int CUresult;
typedef int CUdevice;
typedef void* H;                     // opaque handle (ctx, module, library, kernel, function, stream, event, graph, ...)
typedef unsigned long long CUdeviceptr;
typedef unsigned long long cuuint64_t;
typedef struct { char b[16]; } CUuuid;
#define OK 0
#define E_INVALID_VALUE 1
#define E_OUT_OF_MEMORY 2
#define E_NOT_INITIALIZED 3
#define E_INSUFFICIENT_DRIVER 35
#define E_INVALID_CONTEXT 201
#define E_INVALID_HANDLE 400
#define E_NOT_FOUND 500
#define E_NOT_READY 600
#define E_NOT_PERMITTED 800
#define E_NOT_SUPPORTED 801
#define E_UNKNOWN 999
#define MAXP 1024

// ---------------- transport ----------------
static wd_page_t* P; static uint8_t* D; static wd_welcome_t WL;
static int sock = -1, efd = -1, inited, lost, init_err;
static uint32_t handles_epoch;
static int sticky_seen;          // first sticky device fault code returned to this process   // worker epoch our cached handles belong to (inherited across fork)
static size_t rpc_size, ring_off, ring_size;
static pthread_mutex_t mu = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
static void unlk(int* x) { (void)x; pthread_mutex_unlock(&mu); }
#define LK pthread_mutex_lock(&mu); __attribute__((cleanup(unlk))) int _lk = 0; (void)_lk
static int ensure_init(void);
#define ENTER LK; { int e_ = ensure_init(); if (e_) return e_; }
#define A uint64_t a[16] = {0}, r[16] = {0}
static int trace, stats, spin_us = 30;
static double ST_T[OP_MAX]; static long ST_N[OP_MAX]; static double ST_HASH; static long ST_HASHB;
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static uint8_t* data(void) { return D; }
static uint64_t hash64(const void* p, size_t n) { const uint64_t* w = p; uint64_t h = 0x9E3779B97F4A7C15ull ^ n; size_t k = n / 8;
  for (size_t i = 0; i < k; i++) { h ^= w[i]; h *= 0xff51afd7ed558ccdull; h ^= h >> 29; }
  const uint8_t* t = (const uint8_t*)(w + k); for (size_t i = 0; i < (n & 7); i++) { h ^= t[i]; h *= 1099511628211ull; } return h; }

static int lose(int code, const char* why) {
  if (!lost) { lost = code; fprintf(stderr, "[warmd] pid %d: GPU context lost (%s); CUDA calls now return %d\n", getpid(), why, code); }
  return lost; }
static int rpc_(int op, uint64_t* a, uint64_t* r) {
  if (lost) return lost;
  if (P->dead) return lose(P->dead, "host reported");
  P->op = op; memcpy(P->a, a, sizeof P->a); __sync_synchronize(); P->state = WD_REQ;
  uint64_t one = 1; if (write(efd, &one, 8) != 8) return lose(WD_ERR_DEVICE_LOST, "doorbell failed");
  double t0 = 0;
  for (int it = 0;; it++) {
    int v = P->state; if (v == WD_DONE) break;
    if (it < 64 || (spin_us && (t0 ? now_us() - t0 : (t0 = now_us(), 0)) < spin_us)) { __builtin_ia32_pause(); continue; }
    struct timespec ts = {0, 200 * 1000000};
    syscall(SYS_futex, (int*)&P->state, FUTEX_WAIT, v, &ts, 0, 0);
    if (P->state == WD_DONE) break;
    if (P->dead) return lose(P->dead, "host reported");
    struct pollfd pf = {sock, POLLIN | POLLRDHUP, 0};
    if (poll(&pf, 1, 0) > 0 && (pf.revents & (POLLHUP | POLLRDHUP | POLLERR))) return lose(WD_ERR_DEVICE_LOST, "host connection closed");
  }
  memcpy(r, P->r, sizeof P->r); int e = P->err; P->state = WD_IDLE;
  if (!sticky_seen && (e == 700 || (e >= 714 && e <= 719) || e == 214)) sticky_seen = e;
  if (P->dead) return lose(P->dead, "host reported");
  if (trace > 1 || (trace && e)) fprintf(stderr, "[warmd] rpc op=%d a0=%#lx -> %d\n", op, (unsigned long)a[0], e);
  return e;
}
static int rpc(int op, uint64_t* a, uint64_t* r) {
  if (!stats) return rpc_(op, a, r);
  double t = now_us(); int e = rpc_(op, a, r); ST_T[op % OP_MAX] += now_us() - t; ST_N[op % OP_MAX]++; return e; }

// cached device info (one RPC per process)
static int info_ok; static uint64_t I[16]; static uint8_t IDATA[4096];
static int* ATTR = (int*)(IDATA + 1024); static int* ATTRERR = (int*)(IDATA + 2048);

static void drop_session(void) {
  if (P) munmap(P, WL.region_size);
  if (sock >= 0) close(sock);
  if (efd >= 0) close(efd);
  P = 0; D = 0; sock = efd = -1; inited = 0; lost = 0; init_err = 0;
}
static void reset_local_state(void);
static void after_fork_child(void) {   // the child gets its own session on its first CUDA call
  pthread_mutex_t m = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP; mu = m;
  drop_session(); reset_local_state(); }

static int connect_host(void) {
  const char* path = getenv("WARMD_SOCKET"); if (!path || !*path) path = "/tmp/warmd/gpu0.sock";
  const char* to = getenv("WARMD_CONNECT_TIMEOUT_MS"); int timeout_ms = to ? atoi(to) : 5000;
  struct sockaddr_un sa = {AF_UNIX}; snprintf(sa.sun_path, sizeof sa.sun_path, "%s", path);
  double t0 = now_us();
  for (;;) {
    sock = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (connect(sock, (struct sockaddr*)&sa, sizeof sa) == 0) break;
    int er = errno; close(sock); sock = -1;
    if ((er != ENOENT && er != ECONNREFUSED) || now_us() - t0 > timeout_ms * 1e3) {
      fprintf(stderr, "[warmd] cannot connect to %s: %s\n", path, strerror(er)); return E_NOT_INITIALIZED; }
    usleep(20000);
  }
  wd_hello_t h = {WD_MAGIC, WD_PROTO_VERSION, (uint32_t)getpid(), 0};
  if (send(sock, &h, sizeof h, MSG_NOSIGNAL) != sizeof h) { drop_session(); return E_NOT_INITIALIZED; }
  int fds[4] = {-1, -1}, nfd = 0;
  struct iovec io = {&WL, sizeof WL}; char cbuf[CMSG_SPACE(sizeof(int) * 4)];
  struct msghdr m = {0}; m.msg_iov = &io; m.msg_iovlen = 1; m.msg_control = cbuf; m.msg_controllen = sizeof cbuf;
  ssize_t n; do n = recvmsg(sock, &m, MSG_CMSG_CLOEXEC); while (n < 0 && errno == EINTR);
  for (struct cmsghdr* c = CMSG_FIRSTHDR(&m); c; c = CMSG_NXTHDR(&m, c))
    if (c->cmsg_type == SCM_RIGHTS) { nfd = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int); memcpy(fds, CMSG_DATA(c), sizeof(int) * (nfd > 4 ? 4 : nfd)); }
  if (n != sizeof WL || WL.magic != WD_MAGIC) { fprintf(stderr, "[warmd] bad welcome from host\n"); drop_session(); return E_UNKNOWN; }
  if (WL.status != WD_OK) {
    fprintf(stderr, "[warmd] host refused session: %s\n", WL.reason);
    int st = WL.status; drop_session();
    // E_FULL / E_NOWORKER are "try again later", not "context lost" (launchers must not quarantine on them)
    return st == WD_E_VERSION ? E_INSUFFICIENT_DRIVER : st == WD_E_PERM ? E_NOT_PERMITTED : st == WD_E_FULL ? E_OUT_OF_MEMORY : E_NOT_INITIALIZED; }
  if (nfd != 2) { drop_session(); return E_UNKNOWN; }
  P = mmap(0, WL.region_size, PROT_READ | PROT_WRITE, MAP_SHARED, fds[0], 0); close(fds[0]); efd = fds[1];
  if (P == MAP_FAILED) { P = 0; drop_session(); return E_OUT_OF_MEMORY; }
  D = WD_DATA(P); rpc_size = WL.rpc_size; ring_off = WL.ring_off; ring_size = WL.ring_size;
  // A forked child inherits handles (modules, functions, streams...) created under its parent's
  // session. If the host failed over since, those handles are invalid: fail fast, do not misbehave.
  if (handles_epoch && handles_epoch != WL.epoch) { inited = 1; lose(WD_ERR_DEVICE_LOST, "handles inherited from before a host failover"); return OK; }
  handles_epoch = WL.epoch;
  return OK;
}
// Returns 0 once connected, even after the context was lost: local queries keep working (as with a
// native sticky error); every call that needs the GPU returns the sticky error from rpc().
static int ensure_init(void) {
  if (inited) return 0;
  if (init_err) return init_err;
  { const char* t = getenv("WARMD_TRACE"); trace = t ? atoi(t) : 0; if (t && *t && !trace) trace = 1; }
  stats = getenv("WARMD_STATS") != 0;
  { const char* s = getenv("WARMD_SPIN_US"); if (s) spin_us = atoi(s); }
  static int af; if (!af) { af = 1; pthread_atfork(0, 0, after_fork_child); }
  int e = connect_host(); if (e) { init_err = e; return e; }
  inited = 1;
  if (!info_ok) { A; if (rpc(OP_DEV_INFO, a, r) == 0) { memcpy(I, r, sizeof I); memcpy(IDATA, data(), sizeof IDATA); info_ok = 1; } }
  // Hidden export tables (dark.c) are reverse-engineered per driver branch. Warn on branches we have not validated.
  static const char* VALIDATED[] = {"570.", 0};
  const char* dv = (const char*)IDATA + 3072; int okv = !*dv;
  for (int k = 0; VALIDATED[k]; k++) if (!strncmp(dv, VALIDATED[k], strlen(VALIDATED[k]))) okv = 1;
  if (!okv && !getenv("WARMD_ALLOW_UNVALIDATED_DRIVER"))
    fprintf(stderr, "[warmd] warning: driver %s is not validated (validated: 570.x); run tests/run.sh, see docs/reference/compatibility.md\n", dv);
  if (trace) fprintf(stderr, "[warmd] session %u, region %.1f MB\n", WL.session_id, WL.region_size / 1048576.0);
  return 0;
}
static void print_stats(void) {
  if (!stats) return; double tot = 0; long n = 0;
  for (int i = 0; i < OP_MAX; i++) { tot += ST_T[i]; n += ST_N[i]; }
  fprintf(stderr, "{\"shim_stats\":1,\"pid\":%d,\"rpc_n\":%ld,\"rpc_ms\":%.2f,\"hash_ms\":%.2f,\"hash_MB\":%.1f,\"ops\":{", getpid(), n, tot / 1e3, ST_HASH / 1e3, ST_HASHB / 1e6);
  int f = 1; for (int i = 0; i < OP_MAX; i++) if (ST_N[i]) { fprintf(stderr, "%s\"%d\":[%ld,%.2f]", f ? "" : ",", i, ST_N[i], ST_T[i] / 1e3); f = 0; }
  fprintf(stderr, "}}\n"); }
// the session ends when the socket closes (exit, crash, or kill): the host frees everything then
__attribute__((destructor)) static void shim_fini(void) { print_stats(); }
// for tools that want stats without a clean exit (e.g. os._exit)
void cuWarmdPrintStats(void) { print_stats(); }
// launcher hook: WD_ERR_DEVICE_LOST if the shared context was lost, else the first sticky fault code
// (e.g. 700) this process saw, else 0. Lets a task runner tell "GPU died" from "my code raised".
int cuWarmdStatus(void) { return lost ? lost : sticky_seen; }

// ---------------- local bookkeeping ----------------
typedef struct { uint64_t p, n; int host; } rng_t;
static rng_t* RG; static int nrg, caprg;
static void rg_add(uint64_t p, uint64_t n, int host) { if (nrg == caprg) { caprg = caprg ? caprg * 2 : 256; RG = realloc(RG, caprg * sizeof *RG); } RG[nrg++] = (rng_t){p, n, host}; }
static void rg_del(uint64_t p) { for (int i = 0; i < nrg; i++) if (RG[i].p == p) { RG[i] = RG[--nrg]; return; } }
static rng_t* rg_find(uint64_t p) { for (int i = nrg - 1; i >= 0; i--) if (p >= RG[i].p && p < RG[i].p + (RG[i].n ? RG[i].n : 1)) return &RG[i]; return 0; }
static int is_dev(uint64_t p) { rng_t* x = rg_find(p); return x && !x->host; }
static int is_pinned(const void* p, size_t n) { rng_t* x = rg_find((uint64_t)p); return x && x->host && (uint64_t)p + n <= x->p + x->n; }

typedef struct { H f; int n; uint32_t* off; uint32_t* sz; } finfo_t;
static finfo_t* FI; static int nfi, capfi;
static finfo_t* finfo(H f) {
  for (int i = 0; i < nfi; i++) if (FI[i].f == f) return &FI[i];
  A; a[0] = (uint64_t)f; if (rpc(OP_PARAMINFO, a, r)) return 0;
  if (nfi == capfi) { capfi = capfi ? capfi * 2 : 256; FI = realloc(FI, capfi * sizeof *FI); }
  finfo_t* x = &FI[nfi++]; x->f = f; x->n = (int)r[0]; x->off = malloc(4 * (x->n + 1)); x->sz = malloc(4 * (x->n + 1));
  uint64_t* p = (uint64_t*)data(); for (int k = 0; k < x->n; k++) { x->off[k] = p[2 * k]; x->sz[k] = p[2 * k + 1]; }
  return x;
}

// async copy ring + deferred device->pinned-host copies
typedef struct { void* dst; size_t off, n; uint64_t ev; } pend_t;
static pend_t* PD; static int npd, cappd; static size_t ring_head;
// capture tracking (so IsCapturing/GetCaptureInfo need no RPC when nothing is capturing)
static H CAPS[64]; static int ncaps;
static void reset_local_state(void) { npd = 0; ring_head = 0; ncaps = 0; }

static int flush_pending(void) {     // copy out every deferred D2H whose copy has completed
  int e = OK;
  while (npd) {
    size_t n = npd; if (n * 9 > rpc_size) n = rpc_size / 9;
    uint64_t* ev = (uint64_t*)data(); for (size_t k = 0; k < n; k++) ev[k] = PD[k].ev;
    A; a[0] = n; e = rpc(OP_PENDING_QUERY, a, r); if (e) return e;
    uint8_t* done = data() + n * 8; int w = 0;
    for (size_t k = 0; k < n; k++) {
      if (done[k] == 1) memcpy(PD[k].dst, D + ring_off + PD[k].off, PD[k].n);
      else if (done[k] == 0) PD[w++] = PD[k];
    }
    size_t moved = n - w;
    memmove(PD + w, PD + n, (npd - n) * sizeof *PD); npd -= moved;
    if (w) break;                    // something still in flight
  }
  if (!npd) ring_head = 0;
  return e;
}
static int ring_alloc(size_t n, size_t* off) {
  n = (n + 255) & ~255ul; if (n > ring_size) return -1;
  if (ring_head + n > ring_size) {   // wrap: wait for all users of the ring, then copy out
    A; int e = rpc(OP_RING_SYNC, a, r); if (e) return e;
    e = flush_pending(); if (e) return e;
    if (npd) return -1;
    ring_head = 0;
  }
  *off = ring_head; ring_head += n; return OK;
}

static __thread H cur; static int retained;
#define FAKE_CTX ((H)(uintptr_t)0xC7C70001u)

// ---------------- init / device ----------------
CUresult cuInit(unsigned f) { (void)f; LK; return ensure_init(); }
CUresult cuDriverGetVersion(int* v) { ENTER; *v = info_ok ? (int)I[0] : 12040; return OK; }
CUresult cuDeviceGetCount(int* c) { ENTER; *c = 1; return OK; }
CUresult cuDeviceGet(CUdevice* d, int ord) { ENTER; if (ord != 0) return 101; *d = 0; return OK; }
CUresult cuDeviceGetName(char* b, int n, CUdevice d) { (void)d; ENTER; snprintf(b, n, "%s", (char*)IDATA); return OK; }
CUresult cuDeviceGetAttribute(int* v, int att, CUdevice d) { (void)d; ENTER; if (att < 0 || att >= 256) return E_INVALID_VALUE;
  if (ATTRERR[att]) return ATTRERR[att]; *v = ATTR[att]; return OK; }
CUresult cuDeviceTotalMem_v2(size_t* b, CUdevice d) { (void)d; ENTER; *b = I[5] ? I[5] : I[1]; return OK; }
CUresult cuDeviceGetUuid(CUuuid* u, CUdevice d) { (void)d; ENTER; memcpy(u, IDATA + 256, 16); return OK; }
CUresult cuDeviceGetUuid_v2(CUuuid* u, CUdevice d) { return cuDeviceGetUuid(u, d); }
CUresult cuDeviceGetPCIBusId(char* b, int n, CUdevice d) { (void)d; ENTER; snprintf(b, n, "%s", (char*)IDATA + 512); return OK; }
CUresult cuDeviceComputeCapability(int* ma, int* mi, CUdevice d) { (void)d; ENTER; *ma = ATTR[75]; *mi = ATTR[76]; return OK; }
CUresult cuDeviceCanAccessPeer(int* c, CUdevice d, CUdevice p) { (void)d; (void)p; *c = 0; return OK; }
CUresult cuDeviceGetLuid(char* l, unsigned* m, CUdevice d) { (void)l; (void)m; (void)d; return E_NOT_SUPPORTED; }

// ---------------- contexts (all local: the only real context lives in the host) ----------------
CUresult cuDevicePrimaryCtxRetain(H* c, CUdevice d) { (void)d; ENTER; retained++; *c = FAKE_CTX; return OK; }
CUresult cuDevicePrimaryCtxRelease_v2(CUdevice d) { (void)d; if (retained) retained--; return OK; }
CUresult cuDevicePrimaryCtxGetState(CUdevice d, unsigned* f, int* act) { (void)d; ENTER; if (f) *f = 0; if (act) *act = retained > 0; return OK; }
CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice d, unsigned f) { (void)d; (void)f; return OK; }
CUresult cuDevicePrimaryCtxReset_v2(CUdevice d) { (void)d; retained = 0; return OK; }
CUresult cuCtxCreate_v2(H* c, unsigned f, CUdevice d) { (void)f; (void)d; ENTER; *c = FAKE_CTX; cur = *c; return OK; }
CUresult cuCtxDestroy_v2(H c) { if (cur == c) cur = 0; return OK; }
CUresult cuCtxSetCurrent(H c) { cur = c; return OK; }
// There is exactly one context. Once the primary is retained, a thread with nothing current sees it
// (native cudart makes it current per thread through internal paths we do not observe).
#define CUR (cur ? cur : (retained ? FAKE_CTX : (H)0))
CUresult cuCtxGetCurrent(H* c) { *c = CUR; return OK; }
CUresult cuCtxPushCurrent_v2(H c) { cur = c; return OK; }
CUresult cuCtxPopCurrent_v2(H* c) { if (c) *c = CUR; return OK; }
CUresult cuCtxGetDevice(CUdevice* d) { if (!CUR) return E_INVALID_CONTEXT; *d = 0; return OK; }
CUresult cuCtxGetFlags(unsigned* f) { *f = 0; return OK; }
CUresult cuCtxGetApiVersion(H c, unsigned* v) { (void)c; *v = 3020; return OK; }
CUresult cuCtxGetId(H c, unsigned long long* id) { (void)c; *id = 1; return OK; }
CUresult cuCtxGetStreamPriorityRange(int* lo, int* hi) { ENTER; if (lo) *lo = (int)I[2]; if (hi) *hi = (int)I[3]; return OK; }
CUresult cuCtxSynchronize(void) { ENTER; A; int e = rpc(OP_CTX_SYNC, a, r); if (!e && npd) e = flush_pending(); return e; }
CUresult cuCtxGetLimit(size_t* v, int lim) { ENTER; A; a[0] = lim; int e = rpc(OP_CTX_LIMIT, a, r); *v = r[0]; return e; }
CUresult cuCtxSetLimit(int lim, size_t v) { ENTER; A; a[0] = lim; a[1] = v; a[2] = 1; return rpc(OP_CTX_LIMIT, a, r); }
CUresult cuCtxGetCacheConfig(int* c) { *c = 0; return OK; }
CUresult cuCtxSetCacheConfig(int c) { (void)c; return OK; }

// ---------------- memory ----------------
CUresult cuMemAlloc_v2(CUdeviceptr* p, size_t n) { ENTER; A; a[0] = n; int e = rpc(OP_MEM_ALLOC, a, r); *p = r[0]; if (!e) rg_add(r[0], n, 0); return e; }
CUresult cuMemFree_v2(CUdeviceptr p) { if (!p) return OK; ENTER; A; a[0] = p; int e = rpc(OP_MEM_FREE, a, r); if (!e) rg_del(p); return e; }
CUresult cuMemGetInfo_v2(size_t* f, size_t* t) { ENTER; A; int e = rpc(OP_MEMINFO, a, r); *f = r[0]; *t = r[1]; return e; }
CUresult cuMemGetAddressRange_v2(CUdeviceptr* b, size_t* n, CUdeviceptr p) { LK; rng_t* x = rg_find(p); if (!x || x->host) return E_NOT_FOUND;
  if (b) *b = x->p; if (n) *n = x->n; return OK; }
CUresult cuMemAllocPitch_v2(CUdeviceptr* p, size_t* pitch, size_t w, size_t h, unsigned esz) { (void)esz;
  size_t pt = (w + 511) & ~511ul; int e = cuMemAlloc_v2(p, pt * h); if (!e) *pitch = pt; return e; }
// stream-ordered allocation, implemented eagerly (alloc now; free after the stream drains)
CUresult cuMemAllocAsync(CUdeviceptr* p, size_t n, H s) { (void)s; return cuMemAlloc_v2(p, n); }
CUresult cuMemFreeAsync(CUdeviceptr p, H s) { ENTER; A; a[0] = (uint64_t)s; int e = rpc(OP_SYNC, a, r); if (e) return e; return cuMemFree_v2(p); }
// pinned host memory is plain local memory, but tracked: async copies to/from it are truly async
CUresult cuMemHostAlloc(void** pp, size_t n, unsigned f) { (void)f; LK; if (posix_memalign(pp, 4096, n ? n : 1)) return E_OUT_OF_MEMORY; rg_add((uint64_t)*pp, n, 1); return OK; }
CUresult cuMemAllocHost_v2(void** pp, size_t n) { return cuMemHostAlloc(pp, n, 0); }
CUresult cuMemFreeHost(void* p) { LK; rg_del((uint64_t)p); free(p); return OK; }
CUresult cuMemHostRegister_v2(void* p, size_t n, unsigned f) { (void)f; LK; rg_add((uint64_t)p, n, 1); return OK; }
CUresult cuMemHostUnregister(void* p) { LK; rg_del((uint64_t)p); return OK; }
CUresult cuMemHostGetFlags(unsigned* f, void* p) { (void)p; *f = 0; return OK; }
CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr* d, void* p, unsigned f) { (void)d; (void)p; (void)f; return E_NOT_SUPPORTED; }

static int h2d_sync(uint64_t dst, const void* src, size_t n, H s) {
  for (size_t o = 0; o < n;) { size_t c = n - o < rpc_size ? n - o : rpc_size; memcpy(data(), (const char*)src + o, c);
    A; a[0] = dst + o; a[1] = c; a[2] = (uint64_t)s; int e = rpc(OP_H2D, a, r); if (e) return e; o += c; } return OK; }
static int d2h_sync(void* dst, uint64_t src, size_t n, H s) {
  for (size_t o = 0; o < n;) { size_t c = n - o < rpc_size ? n - o : rpc_size;
    A; a[0] = src + o; a[1] = c; a[2] = (uint64_t)s; int e = rpc(OP_D2H, a, r); if (e) return e; memcpy((char*)dst + o, data(), c); o += c; } return OK; }
// H->D async: snapshot the source into the ring now, DMA later (stream-ordered). Falls back to sync for big copies.
static int h2d_async(uint64_t dst, const void* src, size_t n, H s) {
  size_t off; if (n > ring_size / 4 || ring_alloc(n, &off)) return h2d_sync(dst, src, n, s);
  memcpy(D + ring_off + off, src, n);
  A; a[0] = dst; a[1] = off; a[2] = n; a[3] = (uint64_t)s; return rpc(OP_H2D_ASYNC, a, r); }
// D->H async: only into pinned memory (CUDA semantics: pageable destinations are synchronous).
static int d2h_async(void* dst, uint64_t src, size_t n, H s) {
  size_t off; if (!is_pinned(dst, n) || n > ring_size / 4 || ring_alloc(n, &off)) return d2h_sync(dst, src, n, s);
  A; a[0] = src; a[1] = off; a[2] = n; a[3] = (uint64_t)s; int e = rpc(OP_D2H_ASYNC, a, r); if (e) return e;
  if (npd == cappd) { cappd = cappd ? cappd * 2 : 64; PD = realloc(PD, cappd * sizeof *PD); }
  PD[npd++] = (pend_t){dst, off, n, r[0]}; return OK; }
static int d2d(uint64_t dst, uint64_t src, size_t n, H s) { A; a[0] = dst; a[1] = src; a[2] = n; a[3] = (uint64_t)s; return rpc(OP_D2D, a, r); }
static int any2any(uint64_t dst, uint64_t src, size_t n, H s, int async) {
  int dd = is_dev(dst), sd = is_dev(src);
  if (dd && sd) return d2d(dst, src, n, s);
  if (dd) return async ? h2d_async(dst, (void*)src, n, s) : h2d_sync(dst, (void*)src, n, s);
  if (sd) return async ? d2h_async((void*)dst, src, n, s) : d2h_sync((void*)dst, src, n, s);
  memcpy((void*)dst, (void*)src, n); return OK; }
CUresult cuMemcpyHtoD_v2(CUdeviceptr d, const void* s, size_t n) { ENTER; return h2d_sync(d, s, n, 0); }
CUresult cuMemcpyDtoH_v2(void* d, CUdeviceptr s, size_t n) { ENTER; return d2h_sync(d, s, n, 0); }
CUresult cuMemcpyDtoD_v2(CUdeviceptr d, CUdeviceptr s, size_t n) { ENTER; return d2d(d, s, n, 0); }
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr d, const void* s, size_t n, H st) { ENTER; return h2d_async(d, s, n, st); }
CUresult cuMemcpyDtoHAsync_v2(void* d, CUdeviceptr s, size_t n, H st) { ENTER; return d2h_async(d, s, n, st); }
CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr d, CUdeviceptr s, size_t n, H st) { ENTER; return d2d(d, s, n, st); }
CUresult cuMemcpy(CUdeviceptr d, CUdeviceptr s, size_t n) { ENTER; return any2any(d, s, n, 0, 0); }
CUresult cuMemcpyAsync(CUdeviceptr d, CUdeviceptr s, size_t n, H st) { ENTER; return any2any(d, s, n, st, 1); }

// 2D copies: CUDA_MEMCPY2D. Host-side rows are packed into the staging area; device side is native.
typedef struct { size_t srcX, srcY; int srcType; const void* srcHost; CUdeviceptr srcDev; void* srcArr; size_t srcPitch;
                 size_t dstX, dstY; int dstType; void* dstHost; CUdeviceptr dstDev; void* dstArr; size_t dstPitch;
                 size_t width, height; } m2d_t;
enum { MT_HOST = 1, MT_DEVICE = 2, MT_ARRAY = 3, MT_UNIFIED = 4 };
static int m2d(const m2d_t* c, H s) {
  ENTER;
  int st = c->srcType, dt = c->dstType;
  if (st == MT_ARRAY || dt == MT_ARRAY) return E_NOT_SUPPORTED;
  if (st == MT_UNIFIED) st = is_dev(c->srcDev) ? MT_DEVICE : MT_HOST;
  if (dt == MT_UNIFIED) dt = is_dev(c->dstDev) ? MT_DEVICE : MT_HOST;
  const char* sh = st == MT_HOST ? (c->srcType == MT_UNIFIED ? (const char*)c->srcDev : (const char*)c->srcHost) : 0;
  char* dh = dt == MT_HOST ? (c->dstType == MT_UNIFIED ? (char*)c->dstDev : (char*)c->dstHost) : 0;
  if (st == MT_HOST && dt == MT_HOST) { for (size_t y = 0; y < c->height; y++) memcpy(dh + (c->dstY + y) * c->dstPitch + c->dstX, sh + (c->srcY + y) * c->srcPitch + c->srcX, c->width); return OK; }
  size_t rows = c->width ? (rpc_size - 256) / c->width : c->height; if (!rows) return E_INVALID_VALUE;
  for (size_t y0 = 0; y0 < c->height; y0 += rows) {
    size_t h = c->height - y0 < rows ? c->height - y0 : rows;
    m2d_t m = *c; m.height = h; m.srcType = st; m.dstType = dt;
    uint8_t* stg = data() + 256;
    if (st == MT_HOST) { for (size_t y = 0; y < h; y++) memcpy(stg + y * c->width, sh + (c->srcY + y0 + y) * c->srcPitch + c->srcX, c->width);
      m.srcX = 0; m.srcY = 0; m.srcPitch = c->width; } else m.srcY = c->srcY + y0;
    if (dt == MT_HOST) { m.dstX = 0; m.dstY = 0; m.dstPitch = c->width; } else m.dstY = c->dstY + y0;
    memcpy(data(), &m, sizeof m);
    A; a[0] = (st == MT_HOST) | ((dt == MT_HOST) << 1); a[1] = (uint64_t)s; int e = rpc(OP_MEMCPY2D, a, r); if (e) return e;
    if (dt == MT_HOST) for (size_t y = 0; y < h; y++) memcpy(dh + (c->dstY + y0 + y) * c->dstPitch + c->dstX, stg + y * c->width, c->width);
  }
  return OK; }
CUresult cuMemcpy2D_v2(const m2d_t* c) { return m2d(c, 0); }
CUresult cuMemcpy2DUnaligned_v2(const m2d_t* c) { return m2d(c, 0); }
CUresult cuMemcpy2DAsync_v2(const m2d_t* c, H s) { return m2d(c, s); }

static int mset(uint64_t p, unsigned v, size_t n, int esz, H s) { ENTER; A; a[0] = p; a[1] = v; a[2] = n; a[3] = esz; a[4] = (uint64_t)s; return rpc(OP_MEMSET, a, r); }
CUresult cuMemsetD8_v2(CUdeviceptr p, unsigned char v, size_t n) { return mset(p, v, n, 1, 0); }
CUresult cuMemsetD16_v2(CUdeviceptr p, unsigned short v, size_t n) { return mset(p, v, n, 2, 0); }
CUresult cuMemsetD32_v2(CUdeviceptr p, unsigned v, size_t n) { return mset(p, v, n, 4, 0); }
CUresult cuMemsetD8Async(CUdeviceptr p, unsigned char v, size_t n, H s) { return mset(p, v, n, 1, s); }
CUresult cuMemsetD16Async(CUdeviceptr p, unsigned short v, size_t n, H s) { return mset(p, v, n, 2, s); }
CUresult cuMemsetD32Async(CUdeviceptr p, unsigned v, size_t n, H s) { return mset(p, v, n, 4, s); }

// pointer attributes (from the local range table)
static int ptr_attr(void* out, int at, uint64_t p, int strict) {
  rng_t* x = rg_find(p);
  if (!x) { if (strict) return E_INVALID_VALUE;
    switch (at) { case 2: *(unsigned*)out = 0; break; case 1: *(H*)out = 0; break; case 9: *(int*)out = -2; break;
      case 3: case 4: *(void**)out = 0; break; default: break; } return OK; }
  switch (at) {
    case 1: *(H*)out = FAKE_CTX; break;
    case 2: *(unsigned*)out = x->host ? 1 : 2; break;
    case 3: if (x->host) return E_INVALID_VALUE; *(uint64_t*)out = p; break;
    case 4: if (!x->host) return E_INVALID_VALUE; *(uint64_t*)out = p; break;
    case 6: case 8: case 10: case 15: *(int*)out = 0; break;
    case 7: *(uint64_t*)out = x->p; break;
    case 9: *(int*)out = 0; break;
    case 11: *(uint64_t*)out = x->p; break;
    case 12: *(size_t*)out = x->n; break;
    case 13: *(int*)out = 1; break;
    default: return E_INVALID_VALUE; }
  return OK; }
CUresult cuPointerGetAttribute(void* o, int at, CUdeviceptr p) { LK; return ptr_attr(o, at, p, 1); }
CUresult cuPointerGetAttributes(unsigned n, int* at, void** o, CUdeviceptr p) { LK; for (unsigned i = 0; i < n; i++) ptr_attr(o[i], at[i], p, 0); return OK; }
CUresult cuPointerSetAttribute(const void* v, int at, CUdeviceptr p) { (void)v; (void)at; (void)p; return OK; }

// ---------------- modules / libraries (shared, keyed code cache on the host) ----------------
static const void* resolve_image(const void* img) {
  const uint32_t* m = img; if (m[0] == 0x466243b1) return *(void* const*)((const char*)img + 8);   // __fatBinC_Wrapper_t
  return img; }
static size_t image_size(const void* img) {
  const uint8_t* b = img; const uint32_t* m = img;
  if (m[0] == 0xBA55ED50) { uint16_t hs = *(const uint16_t*)(b + 6); uint64_t fs = *(const uint64_t*)(b + 8); return hs + fs; }
  if (!memcmp(b, ELFMAG, 4)) { const Elf64_Ehdr* e = img; size_t s1 = e->e_shoff + (size_t)e->e_shnum * e->e_shentsize, s2 = e->e_phoff + (size_t)e->e_phnum * e->e_phentsize; return s1 > s2 ? s1 : s2; }
  return strlen((const char*)img) + 1; }
// Images inside a mapped shared object are keyed by file identity + offset (O(1), exact); others by content.
static uint64_t image_key(const void* img, size_t n) {
  Dl_info di; struct stat st;
  if (!getenv("WARMD_HASH_ALL") && dladdr(img, &di) && di.dli_fname && di.dli_fname[0] == '/' && !stat(di.dli_fname, &st)) {
    uint64_t v[6] = {(uint64_t)st.st_dev, (uint64_t)st.st_ino, (uint64_t)st.st_mtime, (uint64_t)st.st_size, (uint64_t)((const char*)img - (const char*)di.dli_fbase), n};
    return hash64(v, sizeof v) ^ 0x5a5a5a5a5a5a5a5aull; }
  return hash64(img, n); }
static int upload(const void* src, size_t n, int* blob) {
  *blob = 0;
  if (n <= rpc_size) { memcpy(data(), src, n); return OK; }
  *blob = 1;
  for (size_t o = 0; o < n;) { size_t c = n - o < rpc_size ? n - o : rpc_size; memcpy(data(), (const char*)src + o, c);
    A; a[0] = o; a[1] = c; a[2] = n; int e = rpc(OP_BLOB, a, r); if (e) return e; o += c; }
  return OK; }
static int load_image(int kind, const void* img, uint64_t* out) {
  img = resolve_image(img); size_t n = image_size(img); double th = stats ? now_us() : 0; uint64_t h = image_key(img, n);
  if (stats) { ST_HASH += now_us() - th; ST_HASHB += n; }
  { A; a[0] = h; a[1] = n; a[2] = kind; if (rpc(OP_LIB_LOOKUP, a, r) == 0 && r[0]) { *out = r[0]; return OK; } }
  int blob; int e = upload(img, n, &blob); if (e) return e;
  A; a[0] = h; a[1] = n; a[2] = kind; a[3] = blob; e = rpc(OP_LIB_LOAD, a, r); *out = r[0];
  return e; }
CUresult cuLibraryLoadData(H* lib, const void* code, void* jo, void** jv, unsigned nj, void* lo, void** lv, unsigned nl) {
  (void)jo; (void)jv; (void)nj; (void)lo; (void)lv; (void)nl; ENTER; uint64_t h = 0; int e = load_image(0, code, &h); *lib = (H)h; return e; }
CUresult cuModuleLoadData(H* m, const void* img) { ENTER; uint64_t h = 0; int e = load_image(1, img, &h); *m = (H)h; return e; }
CUresult cuModuleLoadDataEx(H* m, const void* img, unsigned n, void* o, void** v) { (void)n; (void)o; (void)v; return cuModuleLoadData(m, img); }
CUresult cuModuleLoadFatBinary(H* m, const void* img) { return cuModuleLoadData(m, img); }
static void* slurp(const char* fn, size_t* np) { FILE* f = fopen(fn, "rb"); if (!f) return 0; fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  char* b = calloc(1, n + 1); if (fread(b, 1, n, f) != (size_t)n) { free(b); b = 0; } fclose(f); if (np) *np = n; return b; }
CUresult cuModuleLoad(H* m, const char* fn) { void* b = slurp(fn, 0); if (!b) return E_NOT_FOUND; int e = cuModuleLoadData(m, b); free(b); return e; }
CUresult cuLibraryLoadFromFile(H* lib, const char* fn, void* jo, void** jv, unsigned nj, void* lo, void** lv, unsigned nl) {
  void* b = slurp(fn, 0); if (!b) return E_NOT_FOUND; int e = cuLibraryLoadData(lib, b, jo, jv, nj, lo, lv, nl); free(b); return e; }
CUresult cuModuleUnload(H m) { (void)m; return OK; }   // code stays cached on the host for the next task
CUresult cuLibraryUnload(H l) { (void)l; return OK; }
CUresult cuModuleGetLoadingMode(int* m) { ENTER; *m = (int)I[4]; return OK; }
static int named(int op, uint64_t h, const char* name, uint64_t* r0, uint64_t* r1) {
  ENTER; snprintf((char*)data(), 4096, "%s", name); A; a[0] = h; int e = rpc(op, a, r); if (r0) *r0 = r[0]; if (r1) *r1 = r[1]; return e; }
CUresult cuModuleGetFunction(H* f, H m, const char* name) { uint64_t x = 0; int e = named(OP_GET_FUNC, (uint64_t)m, name, &x, 0); *f = (H)x; return e; }
CUresult cuLibraryGetKernel(H* k, H l, const char* name) { uint64_t x = 0; int e = named(OP_LIB_GET_KERNEL, (uint64_t)l, name, &x, 0); *k = (H)x; return e; }
CUresult cuModuleGetGlobal_v2(CUdeviceptr* p, size_t* n, H m, const char* name) { uint64_t x = 0, y = 0; int e = named(OP_MOD_GET_GLOBAL, (uint64_t)m, name, &x, &y);
  if (p) *p = x; if (n) *n = y; if (!e) { LK; if (!rg_find(x)) rg_add(x, y, 0); } return e; }
CUresult cuLibraryGetGlobal(CUdeviceptr* p, size_t* n, H l, const char* name) { uint64_t x = 0, y = 0; int e = named(OP_LIB_GET_GLOBAL, (uint64_t)l, name, &x, &y);
  if (p) *p = x; if (n) *n = y; if (!e) { LK; if (!rg_find(x)) rg_add(x, y, 0); } return e; }
CUresult cuLibraryGetModule(H* m, H l) { ENTER; A; a[0] = (uint64_t)l; int e = rpc(OP_LIB_GET_MODULE, a, r); *m = (H)r[0]; return e; }
CUresult cuKernelGetFunction(H* f, H k) { ENTER; A; a[0] = (uint64_t)k; int e = rpc(OP_KERNEL_GET_FUNC, a, r); *f = (H)r[0]; return e; }
CUresult cuFuncGetParamInfo(H f, size_t i, size_t* off, size_t* sz) { ENTER; finfo_t* x = finfo(f);
  if (!x || (int)i >= x->n) return E_INVALID_VALUE; *off = x->off[i]; *sz = x->sz[i]; return OK; }
CUresult cuKernelGetParamInfo(H k, size_t i, size_t* off, size_t* sz) { return cuFuncGetParamInfo(k, i, off, sz); }
CUresult cuFuncGetAttribute(int* v, int at, H f) { ENTER; A; a[0] = at; a[1] = (uint64_t)f; int e = rpc(OP_FUNC_GET_ATTR, a, r); *v = (int)r[0]; return e; }
CUresult cuKernelGetAttribute(int* v, int at, H k, CUdevice d) { (void)d; return cuFuncGetAttribute(v, at, k); }
CUresult cuFuncSetAttribute(H f, int at, int v) { ENTER; A; a[0] = at; a[1] = (uint64_t)f; a[2] = (uint64_t)(int64_t)v; return rpc(OP_FUNC_SET_ATTR, a, r); }
CUresult cuKernelSetAttribute(int at, int v, H k, CUdevice d) { (void)d; return cuFuncSetAttribute(k, at, v); }
CUresult cuFuncSetCacheConfig(H f, int c) { (void)f; (void)c; return OK; }
CUresult cuKernelSetCacheConfig(H k, int c, CUdevice d) { (void)k; (void)c; (void)d; return OK; }
CUresult cuFuncSetSharedMemConfig(H f, int c) { (void)f; (void)c; return OK; }
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* nb, H f, int bs, size_t sm, unsigned fl) {
  ENTER; A; a[0] = (uint64_t)f; a[1] = bs; a[2] = sm; a[3] = fl; int e = rpc(OP_OCCUPANCY, a, r); *nb = (int)r[0]; return e; }
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* nb, H f, int bs, size_t sm) { return cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(nb, f, bs, sm, 0); }

// ---------------- launch ----------------
#define LAUNCH_ATTR_SIZE 72   /* sizeof(CUlaunchAttribute): 4 B id + 4 B pad + 64 B value */
static int launch(H f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned smem, H s,
                  void** params, void** extra, const void* attrs, unsigned nattrs) {
  ENTER; size_t total = 0;
  if (params) { finfo_t* x = finfo(f); if (!x) return E_INVALID_VALUE;
    for (int k = 0; k < x->n; k++) { memcpy(data() + x->off[k], params[k], x->sz[k]); if (x->off[k] + x->sz[k] > total) total = x->off[k] + x->sz[k]; } }
  else if (extra) { void* buf = 0; size_t n = 0;
    for (int k = 0; extra[k]; k += 2) { if (extra[k] == (void*)1) buf = extra[k + 1]; else if (extra[k] == (void*)2) n = *(size_t*)extra[k + 1]; }
    if (buf && n) { memcpy(data(), buf, n); total = n; } }
  if (nattrs) memcpy(data() + ((total + 7) & ~7ul), attrs, (size_t)nattrs * LAUNCH_ATTR_SIZE);
  A; a[0] = (uint64_t)f; a[1] = gx; a[2] = gy; a[3] = gz; a[4] = bx; a[5] = by; a[6] = bz; a[7] = smem; a[8] = (uint64_t)s; a[9] = total; a[10] = nattrs;
  return rpc(OP_LAUNCH, a, r); }
CUresult cuLaunchKernel(H f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned smem, H s, void** p, void** x) {
  return launch(f, gx, gy, gz, bx, by, bz, smem, s, p, x, 0, 0); }
CUresult cuLaunchKernel_ptsz(H f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned smem, H s, void** p, void** x) {
  return launch(f, gx, gy, gz, bx, by, bz, smem, s, p, x, 0, 0); }
typedef struct { unsigned gx, gy, gz, bx, by, bz, smem; H s; void* attrs; unsigned nattrs; } launch_cfg_t;
CUresult cuLaunchKernelEx(const launch_cfg_t* c, H f, void** p, void** x) {
  return launch(f, c->gx, c->gy, c->gz, c->bx, c->by, c->bz, c->smem, c->s, p, x, c->attrs, c->nattrs > 64 ? 64 : c->nattrs); }
CUresult cuLaunchCooperativeKernel(H f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned smem, H s, void** p) {
  uint8_t at[LAUNCH_ATTR_SIZE] = {0}; *(int*)at = 2; *(int*)(at + 8) = 1;   // CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2
  return launch(f, gx, gy, gz, bx, by, bz, smem, s, p, 0, at, 1); }

// ---------------- streams / events ----------------
CUresult cuStreamCreate(H* s, unsigned f) { ENTER; A; a[0] = f; int e = rpc(OP_STREAM_CREATE, a, r); *s = (H)r[0]; return e; }
CUresult cuStreamCreateWithPriority(H* s, unsigned f, int p) { ENTER; A; a[0] = f; a[1] = (uint64_t)(int64_t)p; int e = rpc(OP_STREAM_CREATE, a, r); *s = (H)r[0]; return e; }
CUresult cuStreamDestroy_v2(H s) { ENTER; A; a[0] = (uint64_t)s; return rpc(OP_STREAM_DESTROY, a, r); }
CUresult cuStreamSynchronize(H s) { ENTER; A; a[0] = (uint64_t)s; int e = rpc(OP_SYNC, a, r); if (!e && npd) e = flush_pending(); return e; }
CUresult cuStreamQuery(H s) { ENTER; A; a[0] = (uint64_t)s; int e = rpc(OP_STREAM_QUERY, a, r); if (!e && npd) e = flush_pending(); return e; }
CUresult cuStreamWaitEvent(H s, H ev, unsigned f) { ENTER; A; a[0] = (uint64_t)s; a[1] = (uint64_t)ev; a[2] = f; return rpc(OP_STREAM_WAIT_EVENT, a, r); }
CUresult cuStreamGetPriority(H s, int* p) { (void)s; *p = 0; return OK; }
CUresult cuStreamGetFlags(H s, unsigned* f) { (void)s; *f = 0; return OK; }
CUresult cuStreamGetCtx(H s, H* c) { (void)s; *c = FAKE_CTX; return OK; }
CUresult cuStreamGetId(H s, unsigned long long* id) { *id = (uint64_t)s; return OK; }
static int capinfo(H s, int* st, unsigned long long* id) {
  ENTER; if (!ncaps) { *st = 0; if (id) *id = 0; return OK; }
  A; a[0] = (uint64_t)s; int e = rpc(OP_CAPTURE_INFO, a, r); *st = (int)r[0]; if (id) *id = r[1]; return e; }
CUresult cuStreamIsCapturing(H s, int* st) { return capinfo(s, st, 0); }
CUresult cuStreamGetCaptureInfo(H s, int* st, unsigned long long* id) { return capinfo(s, st, id); }
CUresult cuStreamGetCaptureInfo_v2(H s, int* st, unsigned long long* id, H* g, const void** deps, size_t* nd) {
  int e = capinfo(s, st, id); if (g) *g = 0; if (deps) *deps = 0; if (nd) *nd = 0; return e; }
CUresult cuStreamGetCaptureInfo_v3(H s, int* st, unsigned long long* id, H* g, const void** deps, const void** edges, size_t* nd) {
  int e = capinfo(s, st, id); if (g) *g = 0; if (deps) *deps = 0; if (edges) *edges = 0; if (nd) *nd = 0; return e; }
CUresult cuThreadExchangeStreamCaptureMode(int* m) { (void)m; return OK; }
CUresult cuStreamBeginCapture_v2(H s, int mode) { (void)mode; ENTER; A; a[0] = (uint64_t)s; int e = rpc(OP_CAPTURE_BEGIN, a, r);
  if (!e && ncaps < 64) CAPS[ncaps++] = s; return e; }
CUresult cuStreamBeginCapture(H s) { return cuStreamBeginCapture_v2(s, 0); }
CUresult cuStreamEndCapture(H s, H* g) { ENTER; A; a[0] = (uint64_t)s; int e = rpc(OP_CAPTURE_END, a, r); *g = (H)r[0];
  for (int i = 0; i < ncaps; i++) if (CAPS[i] == s) { CAPS[i] = CAPS[--ncaps]; break; } return e; }
CUresult cuGraphInstantiateWithFlags(H* ex, H g, unsigned long long fl) { ENTER; A; a[0] = (uint64_t)g; a[1] = fl; int e = rpc(OP_GRAPH_INSTANTIATE, a, r); *ex = (H)r[0]; return e; }
CUresult cuGraphInstantiate_v2(H* ex, H g, H* errnode, char* log, size_t n) { if (errnode) *errnode = 0; if (log && n) log[0] = 0; return cuGraphInstantiateWithFlags(ex, g, 0); }
typedef struct { unsigned long long flags; H upload; H errNode; int result; } ginst_params_t;
CUresult cuGraphInstantiateWithParams(H* ex, H g, ginst_params_t* p) { int e = cuGraphInstantiateWithFlags(ex, g, p ? p->flags : 0);
  if (p) { p->errNode = 0; p->result = e ? 1 : 0; } return e; }
CUresult cuGraphGetNodes(H g, H* nodes, size_t* n) { ENTER; A; a[0] = (uint64_t)g; a[1] = nodes ? *n : 0; int e = rpc(OP_GRAPH_NODES, a, r);
  if (!e && nodes) memcpy(nodes, data(), r[0] * 8); *n = r[0]; return e; }
CUresult cuGraphLaunch(H ex, H s) { ENTER; A; a[0] = (uint64_t)ex; a[1] = (uint64_t)s; return rpc(OP_GRAPH_LAUNCH, a, r); }
CUresult cuGraphUpload(H ex, H s) { (void)ex; (void)s; return OK; }
CUresult cuGraphExecDestroy(H ex) { ENTER; A; a[0] = (uint64_t)ex; return rpc(OP_GRAPH_EXEC_DESTROY, a, r); }
CUresult cuGraphDestroy(H g) { ENTER; A; a[0] = (uint64_t)g; return rpc(OP_GRAPH_DESTROY, a, r); }

typedef void (*hostfn_t)(void*);
CUresult cuLaunchHostFunc(H s, hostfn_t fn, void* u) { int e = cuStreamSynchronize(s); if (e) return e; fn(u); return OK; }
typedef void (*streamcb_t)(H, CUresult, void*);
CUresult cuStreamAddCallback(H s, streamcb_t cb, void* u, unsigned f) { (void)f; int e = cuStreamSynchronize(s); cb(s, e, u); return OK; }
CUresult cuEventCreate(H* ev, unsigned f) { ENTER; A; a[0] = f; int x = rpc(OP_EVENT_CREATE, a, r); *ev = (H)r[0]; return x; }
CUresult cuEventDestroy_v2(H ev) { ENTER; A; a[0] = (uint64_t)ev; return rpc(OP_EVENT_DESTROY, a, r); }
CUresult cuEventRecord(H ev, H s) { ENTER; A; a[0] = (uint64_t)ev; a[1] = (uint64_t)s; return rpc(OP_EVENT_RECORD, a, r); }
CUresult cuEventRecordWithFlags(H ev, H s, unsigned f) { (void)f; return cuEventRecord(ev, s); }
CUresult cuEventSynchronize(H ev) { ENTER; A; a[0] = (uint64_t)ev; int e = rpc(OP_EVENT_SYNC, a, r); if (!e && npd) e = flush_pending(); return e; }
CUresult cuEventQuery(H ev) { ENTER; A; a[0] = (uint64_t)ev; int e = rpc(OP_EVENT_QUERY, a, r); if (!e && npd) e = flush_pending(); return e; }
CUresult cuEventElapsedTime(float* ms, H e0, H e1) { ENTER; A; a[0] = (uint64_t)e0; a[1] = (uint64_t)e1; int e = rpc(OP_EVENT_ELAPSED, a, r); memcpy(ms, &r[0], 4); return e; }

// ---------------- JIT linking (cuLink*); options are dropped (they carry client-side pointers) ----------------
typedef struct { H st; void* cubin; } link_t; static link_t LINKS[64]; static int nlinks;
CUresult cuLinkCreate_v2(unsigned n, void* o, void** v, H* st) { (void)n; (void)o; (void)v; ENTER; A; int e = rpc(OP_LINK_CREATE, a, r); *st = (H)r[0];
  if (!e && nlinks < 64) LINKS[nlinks++] = (link_t){*st, 0}; return e; }
CUresult cuLinkAddData_v2(H st, int type, void* d, size_t n, const char* name, unsigned no, void* o, void** v) {
  (void)no; (void)o; (void)v; ENTER;
  int blob = 0;
  if (n + 256 <= rpc_size) memcpy(data() + 256, d, n);
  else { int e = upload(d, n, &blob); if (e) return e; }
  snprintf((char*)data(), 256, "%s", name ? name : "");
  A; a[0] = (uint64_t)st; a[1] = type; a[2] = n; a[3] = blob; return rpc(OP_LINK_ADD, a, r); }
CUresult cuLinkAddFile_v2(H st, int type, const char* path, unsigned no, void* o, void** v) {
  size_t n = 0; void* b = slurp(path, &n); if (!b) return E_NOT_FOUND;
  int e = cuLinkAddData_v2(st, type, b, n + (type == 1 /* PTX */ ? 1 : 0), path, no, o, v); free(b); return e; }
CUresult cuLinkComplete(H st, void** cubin, size_t* sz) { ENTER; A; a[0] = (uint64_t)st; int e = rpc(OP_LINK_COMPLETE, a, r); if (e) return e;
  void* b = malloc(r[0]); memcpy(b, data(), r[0]); *cubin = b; *sz = r[0];
  for (int i = 0; i < nlinks; i++) if (LINKS[i].st == st) { free(LINKS[i].cubin); LINKS[i].cubin = b; } return OK; }
CUresult cuLinkDestroy(H st) { ENTER; A; a[0] = (uint64_t)st;
  for (int i = 0; i < nlinks; i++) if (LINKS[i].st == st) { free(LINKS[i].cubin); LINKS[i] = LINKS[--nlinks]; break; }
  return rpc(OP_LINK_DESTROY, a, r); }

// ---------------- misc ----------------
CUresult cuProfilerStart(void) { return OK; }
CUresult cuProfilerStop(void) { return OK; }
#include "dark.c"
CUresult cuGetErrorString(CUresult e, const char** s) {
  static char b[64];
  switch (e) { case 0: *s = "no error"; break; case 1: *s = "invalid argument"; break; case 2: *s = "out of memory"; break;
    case 3: *s = "initialization error (is warmd running?)"; break; case 35: *s = "host protocol version mismatch"; break;
    case WD_ERR_DEVICE_LOST: *s = "GPU context lost (warmd failed over); retry the task"; break;
    case 201: *s = "invalid device context"; break; case 400: *s = "invalid resource handle"; break; case 500: *s = "named symbol not found"; break;
    case 600: *s = "device not ready"; break; case 700: *s = "an illegal memory access was encountered"; break;
    case 800: *s = "operation not permitted"; break; case 801: *s = "operation not supported"; break;
    default: snprintf(b, 64, "CUDA error %d", e); *s = b; }
  return OK; }
CUresult cuGetErrorName(CUresult e, const char** s) {
  switch (e) { case 0: *s = "CUDA_SUCCESS"; return OK; case 1: *s = "CUDA_ERROR_INVALID_VALUE"; return OK; case 2: *s = "CUDA_ERROR_OUT_OF_MEMORY"; return OK;
    case 3: *s = "CUDA_ERROR_NOT_INITIALIZED"; return OK; case WD_ERR_DEVICE_LOST: *s = "CUDA_ERROR_DEVICES_UNAVAILABLE"; return OK;
    case 700: *s = "CUDA_ERROR_ILLEGAL_ADDRESS"; return OK; case 801: *s = "CUDA_ERROR_NOT_SUPPORTED"; return OK; }
  return cuGetErrorString(e, s); }

// ---------------- logging stubs for symbols requested via cuGetProcAddress that we do not implement ----------------
static const char* UNK[400]; static int nunk;
static CUresult unk_call(int i) { fprintf(stderr, "[warmd] UNIMPLEMENTED %s called -> NOT_SUPPORTED\n", UNK[i]); return E_NOT_SUPPORTED; }
#define ST(i) static CUresult stub_##i(void) { return unk_call(i); }
#define D10(p) ST(p##0) ST(p##1) ST(p##2) ST(p##3) ST(p##4) ST(p##5) ST(p##6) ST(p##7) ST(p##8) ST(p##9)
ST(0) ST(1) ST(2) ST(3) ST(4) ST(5) ST(6) ST(7) ST(8) ST(9)
D10(1) D10(2) D10(3) D10(4) D10(5) D10(6) D10(7) D10(8) D10(9) D10(10) D10(11) D10(12) D10(13) D10(14) D10(15) D10(16) D10(17) D10(18) D10(19)
D10(20) D10(21) D10(22) D10(23) D10(24) D10(25) D10(26) D10(27) D10(28) D10(29) D10(30) D10(31) D10(32) D10(33) D10(34) D10(35) D10(36) D10(37) D10(38) D10(39)
#undef ST
#define ST(i) stub_##i,
static CUresult (*STUB[400])(void) = { ST(0) ST(1) ST(2) ST(3) ST(4) ST(5) ST(6) ST(7) ST(8) ST(9)
D10(1) D10(2) D10(3) D10(4) D10(5) D10(6) D10(7) D10(8) D10(9) D10(10) D10(11) D10(12) D10(13) D10(14) D10(15) D10(16) D10(17) D10(18) D10(19)
D10(20) D10(21) D10(22) D10(23) D10(24) D10(25) D10(26) D10(27) D10(28) D10(29) D10(30) D10(31) D10(32) D10(33) D10(34) D10(35) D10(36) D10(37) D10(38) D10(39) };
#undef ST

// ---------------- symbol table: cudart resolves by *base* name + requested version ----------------
CUresult cuGetProcAddress_v2(const char* sym, void** pfn, int ver, cuuint64_t fl, int* st);
CUresult cuGetProcAddress(const char* sym, void** pfn, int ver, cuuint64_t fl);
typedef struct { const char* n; void* f; } sym_t;
#define S(n) {#n, (void*)n},
#define SV(n, f) {#n, (void*)f},
static const sym_t SYMS[] = {
  S(cuInit) S(cuDriverGetVersion) S(cuDeviceGetCount) S(cuDeviceGet) S(cuDeviceGetName) S(cuDeviceGetAttribute)
  SV(cuDeviceTotalMem, cuDeviceTotalMem_v2) SV(cuDeviceGetUuid, cuDeviceGetUuid) S(cuDeviceGetPCIBusId) S(cuDeviceComputeCapability)
  S(cuDeviceCanAccessPeer) S(cuDeviceGetLuid)
  S(cuDevicePrimaryCtxRetain) SV(cuDevicePrimaryCtxRelease, cuDevicePrimaryCtxRelease_v2) S(cuDevicePrimaryCtxGetState)
  SV(cuDevicePrimaryCtxSetFlags, cuDevicePrimaryCtxSetFlags_v2) SV(cuDevicePrimaryCtxReset, cuDevicePrimaryCtxReset_v2)
  SV(cuCtxCreate, cuCtxCreate_v2) SV(cuCtxDestroy, cuCtxDestroy_v2) S(cuCtxSetCurrent) S(cuCtxGetCurrent)
  SV(cuCtxPushCurrent, cuCtxPushCurrent_v2) SV(cuCtxPopCurrent, cuCtxPopCurrent_v2) S(cuCtxGetDevice) S(cuCtxGetFlags)
  S(cuCtxGetApiVersion) S(cuCtxGetId) S(cuCtxGetStreamPriorityRange) S(cuCtxSynchronize) S(cuCtxGetLimit) S(cuCtxSetLimit)
  S(cuCtxGetCacheConfig) S(cuCtxSetCacheConfig)
  SV(cuMemAlloc, cuMemAlloc_v2) SV(cuMemFree, cuMemFree_v2) SV(cuMemGetInfo, cuMemGetInfo_v2) SV(cuMemGetAddressRange, cuMemGetAddressRange_v2)
  SV(cuMemAllocPitch, cuMemAllocPitch_v2) S(cuMemAllocAsync) S(cuMemFreeAsync)
  S(cuMemHostAlloc) SV(cuMemAllocHost, cuMemAllocHost_v2) S(cuMemFreeHost) SV(cuMemHostRegister, cuMemHostRegister_v2)
  S(cuMemHostUnregister) S(cuMemHostGetFlags) SV(cuMemHostGetDevicePointer, cuMemHostGetDevicePointer_v2)
  SV(cuMemcpyHtoD, cuMemcpyHtoD_v2) SV(cuMemcpyDtoH, cuMemcpyDtoH_v2) SV(cuMemcpyDtoD, cuMemcpyDtoD_v2)
  SV(cuMemcpyHtoDAsync, cuMemcpyHtoDAsync_v2) SV(cuMemcpyDtoHAsync, cuMemcpyDtoHAsync_v2) SV(cuMemcpyDtoDAsync, cuMemcpyDtoDAsync_v2)
  S(cuMemcpy) S(cuMemcpyAsync) SV(cuMemcpy2D, cuMemcpy2D_v2) SV(cuMemcpy2DAsync, cuMemcpy2DAsync_v2) SV(cuMemcpy2DUnaligned, cuMemcpy2DUnaligned_v2)
  SV(cuMemsetD8, cuMemsetD8_v2) SV(cuMemsetD16, cuMemsetD16_v2) SV(cuMemsetD32, cuMemsetD32_v2)
  S(cuMemsetD8Async) S(cuMemsetD16Async) S(cuMemsetD32Async)
  S(cuPointerGetAttribute) S(cuPointerGetAttributes) S(cuPointerSetAttribute)
  S(cuLibraryLoadData) S(cuLibraryLoadFromFile) S(cuLibraryUnload) S(cuLibraryGetKernel) S(cuLibraryGetModule) S(cuLibraryGetGlobal)
  S(cuKernelGetFunction) S(cuKernelGetAttribute) S(cuKernelSetAttribute) S(cuKernelSetCacheConfig) S(cuKernelGetParamInfo)
  S(cuModuleLoad) S(cuModuleLoadData) S(cuModuleLoadDataEx) S(cuModuleLoadFatBinary) S(cuModuleUnload) S(cuModuleGetFunction)
  SV(cuModuleGetGlobal, cuModuleGetGlobal_v2) S(cuModuleGetLoadingMode)
  S(cuFuncGetParamInfo) S(cuFuncGetAttribute) S(cuFuncSetAttribute) S(cuFuncSetCacheConfig) S(cuFuncSetSharedMemConfig)
  S(cuOccupancyMaxActiveBlocksPerMultiprocessor) S(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
  S(cuLaunchKernel) S(cuLaunchKernelEx) S(cuLaunchCooperativeKernel) S(cuLaunchHostFunc) S(cuStreamAddCallback)
  S(cuStreamCreate) S(cuStreamCreateWithPriority) SV(cuStreamDestroy, cuStreamDestroy_v2) S(cuStreamSynchronize) S(cuStreamQuery)
  S(cuStreamWaitEvent) S(cuStreamGetPriority) S(cuStreamGetFlags) S(cuStreamGetCtx) S(cuStreamGetId) S(cuStreamIsCapturing)
  S(cuThreadExchangeStreamCaptureMode) SV(cuStreamBeginCapture, cuStreamBeginCapture_v2) S(cuStreamEndCapture)
  S(cuGraphGetNodes) S(cuGraphInstantiateWithFlags) S(cuGraphInstantiateWithParams) S(cuGraphLaunch) S(cuGraphUpload) S(cuGraphExecDestroy) S(cuGraphDestroy)
  S(cuEventCreate) SV(cuEventDestroy, cuEventDestroy_v2) S(cuEventRecord) S(cuEventRecordWithFlags) S(cuEventSynchronize)
  S(cuEventQuery) S(cuEventElapsedTime)
  SV(cuLinkCreate, cuLinkCreate_v2) SV(cuLinkAddData, cuLinkAddData_v2) SV(cuLinkAddFile, cuLinkAddFile_v2) S(cuLinkComplete) S(cuLinkDestroy)
  S(cuProfilerStart) S(cuProfilerStop) S(cuGetExportTable) S(cuGetErrorString) S(cuGetErrorName)
  {0, 0}};
static void* lookup(const char* sym, int ver) {
  // symbols whose signature changed between versions
  if (!strcmp(sym, "cuGetProcAddress")) return ver >= 12000 ? (void*)cuGetProcAddress_v2 : (void*)cuGetProcAddress;
  if (!strcmp(sym, "cuStreamGetCaptureInfo")) return ver >= 12030 ? (void*)cuStreamGetCaptureInfo_v3 : ver >= 11030 ? (void*)cuStreamGetCaptureInfo_v2 : (void*)cuStreamGetCaptureInfo;
  if (!strcmp(sym, "cuGraphInstantiate")) return ver >= 12000 ? (void*)cuGraphInstantiateWithFlags : (void*)cuGraphInstantiate_v2;
  for (const sym_t* s = SYMS; s->n; s++) if (!strcmp(s->n, sym)) return s->f;
  size_t L = strlen(sym);   // explicit _v2/_ptsz/_ptds names
  for (const sym_t* s = SYMS; s->n; s++) { size_t l = strlen(s->n); if (L > l && !strncmp(s->n, sym, l) && sym[l] == '_') return s->f; }
  return 0; }
CUresult cuGetProcAddress_v2(const char* sym, void** pfn, int ver, cuuint64_t fl, int* st) {
  (void)fl; LK; void* f = lookup(sym, ver);
  if (!f) {
    if (getenv("WARMD_UNK_NOTFOUND") || nunk >= 400) { if (st) *st = 1; *pfn = 0; return E_NOT_FOUND; }
    for (int i = 0; i < nunk; i++) if (!strcmp(UNK[i], sym)) { f = (void*)STUB[i]; break; }
    if (!f) { UNK[nunk] = strdup(sym); f = (void*)STUB[nunk]; nunk++; }
    if (trace) fprintf(stderr, "[warmd] resolve %s (v%d) -> logging stub\n", sym, ver);
  } else if (trace > 1) fprintf(stderr, "[warmd] resolve %s (v%d)\n", sym, ver);
  *pfn = f; if (st) *st = 0; return OK; }
CUresult cuGetProcAddress(const char* sym, void** pfn, int ver, cuuint64_t fl) { return cuGetProcAddress_v2(sym, pfn, ver, fl, 0); }

// everything else the real driver exports lives in stubs.c as *weak* stubs/aliases (cuBLASLt dlsyms a fixed list)
__attribute__((visibility("hidden"))) CUresult wd_xstub(const char* n) {
  fprintf(stderr, "[warmd] UNIMPLEMENTED %s called -> NOT_SUPPORTED\n", n); return E_NOT_SUPPORTED; }

// ---------------- exported plain / versioned aliases ----------------
CUresult cuDeviceTotalMem(size_t* b, CUdevice d) { return cuDeviceTotalMem_v2(b, d); }
CUresult cuDevicePrimaryCtxRelease(CUdevice d) { return cuDevicePrimaryCtxRelease_v2(d); }
CUresult cuDevicePrimaryCtxRetain_v2(H* c, CUdevice d) { return cuDevicePrimaryCtxRetain(c, d); }
CUresult cuCtxCreate(H* c, unsigned f, CUdevice d) { return cuCtxCreate_v2(c, f, d); }
CUresult cuCtxDestroy(H c) { return cuCtxDestroy_v2(c); }
CUresult cuCtxPushCurrent(H c) { return cuCtxPushCurrent_v2(c); }
CUresult cuCtxPopCurrent(H* c) { return cuCtxPopCurrent_v2(c); }
CUresult cuMemAlloc(CUdeviceptr* p, size_t n) { return cuMemAlloc_v2(p, n); }
CUresult cuMemFree(CUdeviceptr p) { return cuMemFree_v2(p); }
CUresult cuMemGetInfo(size_t* f, size_t* t) { return cuMemGetInfo_v2(f, t); }
CUresult cuMemsetD8(CUdeviceptr p, unsigned char v, size_t n) { return cuMemsetD8_v2(p, v, n); }
CUresult cuMemsetD32(CUdeviceptr p, unsigned v, size_t n) { return cuMemsetD32_v2(p, v, n); }
CUresult cuMemcpyHtoD(CUdeviceptr d, const void* s, size_t n) { return cuMemcpyHtoD_v2(d, s, n); }
CUresult cuMemcpyDtoH(void* d, CUdeviceptr s, size_t n) { return cuMemcpyDtoH_v2(d, s, n); }
CUresult cuMemcpyDtoD(CUdeviceptr d, CUdeviceptr s, size_t n) { return cuMemcpyDtoD_v2(d, s, n); }
CUresult cuMemcpyHtoDAsync(CUdeviceptr d, const void* s, size_t n, H st) { return cuMemcpyHtoDAsync_v2(d, s, n, st); }
CUresult cuMemcpyDtoHAsync(void* d, CUdeviceptr s, size_t n, H st) { return cuMemcpyDtoHAsync_v2(d, s, n, st); }
CUresult cuMemAllocHost(void** pp, size_t n) { return cuMemAllocHost_v2(pp, n); }
CUresult cuMemHostRegister(void* p, size_t n, unsigned f) { return cuMemHostRegister_v2(p, n, f); }
CUresult cuModuleGetGlobal(CUdeviceptr* p, size_t* n, H m, const char* name) { return cuModuleGetGlobal_v2(p, n, m, name); }
CUresult cuStreamDestroy(H s) { return cuStreamDestroy_v2(s); }
CUresult cuEventDestroy(H e) { return cuEventDestroy_v2(e); }
