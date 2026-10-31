// worker.c: the process that owns the CUDA context (warmd --worker).
//
// Maps and pins every session region at start. Once promoted, N threads wait on one epoll set
// with EPOLLONESHOT, so each session is served by one thread at a time.
#include "common.h"
#include <cuda.h>
#include <pthread.h>
#include <signal.h>
#include <sys/epoll.h>
#include <sys/mman.h>

extern CUresult cuFuncGetParamInfo(CUfunction f, size_t i, size_t* off, size_t* sz);  // driver >= 12.4

#define CKX(x) do { CUresult r_ = (x); if (r_) { const char* s_; cuGetErrorString(r_, &s_); LOGE("%s -> %s", #x, s_); _exit(2); } } while (0)

static int ctl = -1, gpu = 0, nthreads = 4;
static CUcontext ctx; static CUdevice dev;
static wd_stats_t* ST; static wd_stats_t st_dummy;
static int epfd;

// ---------------- per-session state ----------------
enum { RS_MEM, RS_STREAM, RS_EVENT, RS_GRAPH, RS_GEXEC, RS_LINK, RS_N };
typedef struct res_s { uint64_t h; size_t n; int flag; CUevent ev; struct res_s* next; } res_t;
typedef struct {
  pthread_mutex_t mu; pthread_cond_t cv;
  int attached, open; uint32_t sid; uint64_t quota, used;
  wd_page_t* page; uint8_t* data; int efd; size_t size;
  CUstream s0;                 // the session's own default stream (tenant's stream 0 / legacy / per-thread)
  CUevent ev_def;              // used to order blocking streams after the default stream
  res_t* res[RS_N];
  CUevent evpool[64]; int nevpool;   // completion events for deferred D2H copies
  uint8_t* blob; size_t blobn;
} sess_t;
static sess_t S[WD_MAX_SESSIONS]; static int nsess;
static size_t rpc_size, ring_off, ring_size;

static void track(sess_t* s, int t, uint64_t h, size_t n, int flag) {
  res_t* r = calloc(1, sizeof *r); r->h = h; r->n = n; r->flag = flag; r->next = s->res[t]; s->res[t] = r; }
static res_t* find(sess_t* s, int t, uint64_t h) { for (res_t* r = s->res[t]; r; r = r->next) if (r->h == h) return r; return 0; }
static int untrack(sess_t* s, int t, uint64_t h) {
  for (res_t** q = &s->res[t]; *q; q = &(*q)->next) if ((*q)->h == h) { res_t* x = *q; *q = x->next; if (x->ev) cuEventDestroy(x->ev); free(x); return 1; }
  return 0; }

// ---------------- shared code cache (content/file keyed; reused across sessions) ----------------
static pthread_mutex_t cmu = PTHREAD_MUTEX_INITIALIZER;
typedef struct { uint64_t h, n; int kind; uint64_t handle; } lib_t;
static lib_t LIBS[8192]; static int nlibs;
typedef struct { uint64_t k; CUfunction f; } kf_t;
static kf_t KF[1 << 16]; static int nkf;
static void note_kernel(uint64_t k) { pthread_mutex_lock(&cmu); for (int j = 0; j < nkf; j++) if (KF[j].k == k) goto out;
  if (nkf < (1 << 16)) { KF[nkf].k = k; KF[nkf].f = 0; nkf++; } out: pthread_mutex_unlock(&cmu); }
static CUfunction asfunc(uint64_t h) {
  CUfunction f = (CUfunction)h; pthread_mutex_lock(&cmu);
  for (int j = 0; j < nkf; j++) if (KF[j].k == h) { if (!KF[j].f) cuKernelGetFunction(&KF[j].f, (CUkernel)h); f = KF[j].f; break; }
  pthread_mutex_unlock(&cmu); return f; }

// ---------------- streams ----------------
static CUstream smap(sess_t* s, uint64_t h) { return (h == 0 || h == 1 || h == 2) ? s->s0 : (CUstream)h; }
// Emulate legacy default-stream semantics *within* a session: work on the default stream waits for
// prior work on the session's blocking streams, and work on a blocking stream waits for the default.
static void prep_stream(sess_t* s, uint64_t h) {
  if (!s->res[RS_STREAM]) return;
  CUstream q = smap(s, h);
  if (q == s->s0) {
    for (res_t* r = s->res[RS_STREAM]; r; r = r->next) if (r->flag) {
      if (!r->ev) cuEventCreate(&r->ev, CU_EVENT_DISABLE_TIMING);
      cuEventRecord(r->ev, (CUstream)r->h); cuStreamWaitEvent(s->s0, r->ev, 0); }
  } else { res_t* r = find(s, RS_STREAM, h);
    if (r && r->flag) { if (!s->ev_def) cuEventCreate(&s->ev_def, CU_EVENT_DISABLE_TIMING);
      cuEventRecord(s->ev_def, s->s0); cuStreamWaitEvent(q, s->ev_def, 0); } }
}
static CUresult sync_all(sess_t* s) {
  CUresult z = cuStreamSynchronize(s->s0);
  for (res_t* r = s->res[RS_STREAM]; r && !z; r = r->next) z = cuStreamSynchronize((CUstream)r->h);
  return z; }

static void close_session(sess_t* s) {
  if (!s->open) return;
  cuStreamSynchronize(s->s0);
  for (res_t* r = s->res[RS_STREAM]; r; r = r->next) cuStreamSynchronize((CUstream)r->h);
  for (int t = 0; t < RS_N; t++) {
    while (s->res[t]) {
      res_t* r = s->res[t]; s->res[t] = r->next;
      switch (t) {
        case RS_MEM: cuMemFree((CUdeviceptr)r->h); __sync_fetch_and_sub(&ST->mem_used, r->n); break;
        case RS_STREAM: cuStreamDestroy((CUstream)r->h); break;
        case RS_EVENT: cuEventDestroy((CUevent)r->h); break;
        case RS_GRAPH: cuGraphDestroy((CUgraph)r->h); break;
        case RS_GEXEC: cuGraphExecDestroy((CUgraphExec)r->h); break;
        case RS_LINK: cuLinkDestroy((CUlinkState)r->h); break;
      }
      if (r->ev) cuEventDestroy(r->ev); free(r);
    }
  }
  if (s->ev_def) { cuEventDestroy(s->ev_def); s->ev_def = 0; }
  free(s->blob); s->blob = 0; s->blobn = 0;
  s->open = 0; s->used = 0; ST->sessions_served++;
  size_t fr, tot; if (!cuMemGetInfo(&fr, &tot)) { ST->gpu_mem_free = fr; ST->gpu_mem_total = tot; }
}

static int is_sticky(CUresult r) {
  return r == CUDA_ERROR_ILLEGAL_ADDRESS || r == CUDA_ERROR_LAUNCH_FAILED || r == CUDA_ERROR_ILLEGAL_INSTRUCTION ||
         r == CUDA_ERROR_MISALIGNED_ADDRESS || r == CUDA_ERROR_HARDWARE_STACK_ERROR || r == CUDA_ERROR_INVALID_PC ||
         r == CUDA_ERROR_INVALID_ADDRESS_SPACE || r == CUDA_ERROR_ECC_UNCORRECTABLE; }

// ---------------- dispatch ----------------
static int lib_load(sess_t* s, uint64_t* a, uint64_t* r) {
  const void* img = a[3] ? (const void*)s->blob : (const void*)s->data;
  if (a[3] && (!s->blob || s->blobn != a[1])) return CUDA_ERROR_INVALID_VALUE;
  pthread_mutex_lock(&cmu);
  for (int j = 0; j < nlibs; j++) if (LIBS[j].h == a[0] && LIBS[j].n == a[1] && LIBS[j].kind == (int)a[2]) {
    r[0] = LIBS[j].handle; pthread_mutex_unlock(&cmu); ST->lib_hits++; return 0; }
  CUresult z; uint64_t h = 0;
  if (a[2] == 0) { CUlibrary l = 0; z = cuLibraryLoadData(&l, img, 0, 0, 0, 0, 0, 0); h = (uint64_t)l; }
  else { CUmodule m = 0; z = cuModuleLoadData(&m, img); h = (uint64_t)m; }
  if (!z && nlibs < 8192) LIBS[nlibs++] = (lib_t){a[0], a[1], (int)a[2], h};
  pthread_mutex_unlock(&cmu); ST->lib_miss++;
  if (a[3]) { free(s->blob); s->blob = 0; s->blobn = 0; }
  r[0] = h; return z;
}

static CUresult dispatch(int idx, sess_t* s) {
  wd_page_t* p = s->page; uint64_t* a = p->a; uint64_t* r = p->r; uint8_t* d = s->data; CUresult z = 0;
  uint8_t* ring = d + ring_off;
  #define INRPC(off, n) ((off) <= rpc_size && (n) <= rpc_size - (off))
  #define INRING(off, n) ((off) <= ring_size && (n) <= ring_size - (off))
  memset(r, 0, sizeof p->r);
  switch (p->op) {
    case OP_NOP: break;
    case OP_DEV_INFO: {
      memset(d, 0, 4096); int v = 0; size_t m = 0;
      cuDriverGetVersion(&v); r[0] = v; cuDeviceTotalMem(&m, dev); r[1] = m;
      int lo = 0, hi = 0; cuCtxGetStreamPriorityRange(&lo, &hi); r[2] = (uint64_t)(int64_t)lo; r[3] = (uint64_t)(int64_t)hi;
      CUmoduleLoadingMode lm = CU_MODULE_LAZY_LOADING; cuModuleGetLoadingMode(&lm); r[4] = lm; r[5] = s->quota;
      cuDeviceGetName((char*)d, 255, dev); CUuuid u; cuDeviceGetUuid(&u, dev); memcpy(d + 256, &u, 16);
      cuDeviceGetPCIBusId((char*)d + 512, 64, dev);
      int* att = (int*)(d + 1024); int* aerr = (int*)(d + 2048);
      for (int k = 1; k < 256; k++) { int x = 0; aerr[k] = cuDeviceGetAttribute(&x, (CUdevice_attribute)k, dev); att[k] = x; }
      aerr[0] = 1;
      FILE* f = fopen("/proc/driver/nvidia/version", "r");   // e.g. "... Kernel Module  570.211.01 ..."
      if (f) { char line[256]; if (fgets(line, sizeof line, f)) { char* k = strstr(line, "Module"); if (k) { k += 6; while (*k == ' ' || *k == '\t') k++;
        int j = 0; while (k[j] && k[j] != ' ' && j < 63) { d[3072 + j] = k[j]; j++; } } } fclose(f); }
      break; }
    case OP_MEM_ALLOC: {
      size_t n = a[0] ? a[0] : 1;
      if (s->quota && s->used + n > s->quota) { z = CUDA_ERROR_OUT_OF_MEMORY; break; }
      CUdeviceptr q = 0; z = cuMemAlloc(&q, n); r[0] = q;
      if (!z) { track(s, RS_MEM, q, n, 0); s->used += n; __sync_fetch_and_add(&ST->mem_used, n); }
      break; }
    case OP_MEM_FREE: {
      res_t* x = find(s, RS_MEM, a[0]);
      if (!x) { z = CUDA_ERROR_INVALID_VALUE; break; }             // only your own allocations
      size_t n = x->n; z = cuMemFree((CUdeviceptr)a[0]); untrack(s, RS_MEM, a[0]); s->used -= n; __sync_fetch_and_sub(&ST->mem_used, n);
      break; }
    case OP_MEMINFO: {
      size_t f = 0, t = 0; z = cuMemGetInfo(&f, &t); ST->gpu_mem_free = f; ST->gpu_mem_total = t;
      if (s->quota) { size_t left = s->quota - s->used; r[0] = f < left ? f : left; r[1] = s->quota; } else { r[0] = f; r[1] = t; }
      break; }
    case OP_BLOB: {
      if (!INRPC(0, a[1])) { z = CUDA_ERROR_INVALID_VALUE; break; }
      if (a[0] == 0) { free(s->blob); s->blob = malloc(a[2]); s->blobn = a[2]; }
      if (!s->blob || a[0] + a[1] > s->blobn) { z = CUDA_ERROR_INVALID_VALUE; break; }
      memcpy(s->blob + a[0], d, a[1]); break; }
    case OP_LIB_LOOKUP: {
      pthread_mutex_lock(&cmu);
      for (int j = 0; j < nlibs; j++) if (LIBS[j].h == a[0] && LIBS[j].n == a[1] && LIBS[j].kind == (int)a[2]) { r[0] = LIBS[j].handle; ST->lib_hits++; break; }
      pthread_mutex_unlock(&cmu); break; }
    case OP_LIB_LOAD: z = lib_load(s, a, r); break;
    case OP_LIB_GET_MODULE: { CUmodule m = 0; z = cuLibraryGetModule(&m, (CUlibrary)a[0]); r[0] = (uint64_t)m; break; }
    case OP_LIB_GET_KERNEL: { d[4095] = 0; CUkernel k = 0; z = cuLibraryGetKernel(&k, (CUlibrary)a[0], (const char*)d); r[0] = (uint64_t)k; if (!z) note_kernel(r[0]); break; }
    case OP_KERNEL_GET_FUNC: { CUfunction f = asfunc(a[0]); r[0] = (uint64_t)f; z = f ? 0 : CUDA_ERROR_INVALID_HANDLE; break; }
    case OP_LIB_GET_GLOBAL: { d[4095] = 0; CUdeviceptr q = 0; size_t n = 0; z = cuLibraryGetGlobal(&q, &n, (CUlibrary)a[0], (const char*)d); r[0] = q; r[1] = n; break; }
    case OP_MOD_GET_GLOBAL: { d[4095] = 0; CUdeviceptr q = 0; size_t n = 0; z = cuModuleGetGlobal(&q, &n, (CUmodule)a[0], (const char*)d); r[0] = q; r[1] = n; break; }
    case OP_GET_FUNC: { d[4095] = 0; CUfunction f = 0; z = cuModuleGetFunction(&f, (CUmodule)a[0], (const char*)d); r[0] = (uint64_t)f; break; }
    case OP_PARAMINFO: {
      CUfunction f = asfunc(a[0]); size_t off = 0, sz = 0; uint64_t* c = (uint64_t*)d; int n = 0;
      for (size_t k = 0; k < 1024; k++) { if (cuFuncGetParamInfo(f, k, &off, &sz)) break; c[2 * k] = off; c[2 * k + 1] = sz; n++; }
      r[0] = n; break; }
    case OP_FUNC_GET_ATTR: { int v = 0; z = cuFuncGetAttribute(&v, (CUfunction_attribute)a[0], asfunc(a[1])); r[0] = v; break; }
    case OP_FUNC_SET_ATTR: z = cuFuncSetAttribute(asfunc(a[1]), (CUfunction_attribute)a[0], (int)(int64_t)a[2]); break;
    case OP_OCCUPANCY: { int nb = 0; z = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&nb, asfunc(a[0]), (int)a[1], a[2], (unsigned)a[3]); r[0] = nb; break; }
    case OP_LAUNCH: {
      size_t pb = a[9]; uint64_t nat = a[10];
      if (!INRPC(0, pb) || nat > 64 || !INRPC((pb + 7) & ~7ul, nat * sizeof(CUlaunchAttribute))) { z = CUDA_ERROR_INVALID_VALUE; break; }
      prep_stream(s, a[8]);
      void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, d, CU_LAUNCH_PARAM_BUFFER_SIZE, &pb, CU_LAUNCH_PARAM_END};
      if (!nat) z = cuLaunchKernel((CUfunction)a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], smap(s, a[8]), 0, extra);
      else {
        CUlaunchConfig c = {(unsigned)a[1], (unsigned)a[2], (unsigned)a[3], (unsigned)a[4], (unsigned)a[5], (unsigned)a[6], (unsigned)a[7],
                            smap(s, a[8]), (CUlaunchAttribute*)(d + ((pb + 7) & ~7ul)), (unsigned)nat};
        z = cuLaunchKernelEx(&c, (CUfunction)a[0], 0, extra);
      }
      ST->launches++; break; }
    case OP_MEMSET: { CUstream q = smap(s, a[4]); prep_stream(s, a[4]);
      z = a[3] == 1 ? cuMemsetD8Async(a[0], (unsigned char)a[1], a[2], q) : a[3] == 2 ? cuMemsetD16Async(a[0], (unsigned short)a[1], a[2], q)
                    : cuMemsetD32Async(a[0], (unsigned)a[1], a[2], q); break; }
    case OP_H2D: { if (!INRPC(0, a[1])) { z = CUDA_ERROR_INVALID_VALUE; break; } CUstream q = smap(s, a[2]); prep_stream(s, a[2]);
      z = cuMemcpyHtoDAsync((CUdeviceptr)a[0], d, a[1], q); if (!z) z = cuStreamSynchronize(q); ST->bytes_h2d += a[1]; break; }
    case OP_D2H: { if (!INRPC(0, a[1])) { z = CUDA_ERROR_INVALID_VALUE; break; } CUstream q = smap(s, a[2]); prep_stream(s, a[2]);
      z = cuMemcpyDtoHAsync(d, (CUdeviceptr)a[0], a[1], q); if (!z) z = cuStreamSynchronize(q); ST->bytes_d2h += a[1]; if (a[1] > p->hiwater) p->hiwater = a[1]; break; }
    case OP_D2D: prep_stream(s, a[3]); z = cuMemcpyDtoDAsync((CUdeviceptr)a[0], (CUdeviceptr)a[1], a[2], smap(s, a[3])); break;
    case OP_H2D_ASYNC: { if (!INRING(a[1], a[2])) { z = CUDA_ERROR_INVALID_VALUE; break; } prep_stream(s, a[3]);
      z = cuMemcpyHtoDAsync((CUdeviceptr)a[0], ring + a[1], a[2], smap(s, a[3])); ST->bytes_h2d += a[2]; break; }
    case OP_D2H_ASYNC: { if (!INRING(a[1], a[2])) { z = CUDA_ERROR_INVALID_VALUE; break; } CUstream q = smap(s, a[3]); prep_stream(s, a[3]);
      z = cuMemcpyDtoHAsync(ring + a[1], (CUdeviceptr)a[0], a[2], q); if (z) break;
      CUevent e = s->nevpool ? s->evpool[--s->nevpool] : 0; if (!e) cuEventCreate(&e, CU_EVENT_DISABLE_TIMING);
      z = cuEventRecord(e, q); r[0] = (uint64_t)e; track(s, RS_EVENT, (uint64_t)e, 0, 1); ST->bytes_d2h += a[2]; break; }
    case OP_PENDING_QUERY: {   // data: u64 events[a0] -> data: u8 done[a0] (after the u64 array); completed events are recycled
      uint64_t n = a[0]; if (!INRPC(0, n * 9)) { z = CUDA_ERROR_INVALID_VALUE; break; }
      uint64_t* ev = (uint64_t*)d; uint8_t* done = d + n * 8; uint64_t c = 0;
      for (uint64_t k = 0; k < n; k++) {
        res_t* x = find(s, RS_EVENT, ev[k]); if (!x || !x->flag) { done[k] = 2; continue; }
        CUresult q = cuEventQuery((CUevent)ev[k]);
        if (q == CUDA_SUCCESS) { done[k] = 1; c++; x->h = 0; untrack(s, RS_EVENT, 0);
          if (s->nevpool < 64) s->evpool[s->nevpool++] = (CUevent)ev[k]; else cuEventDestroy((CUevent)ev[k]); }
        else if (q == CUDA_ERROR_NOT_READY) done[k] = 0; else { done[k] = 3; z = q; }
      }
      r[0] = c; break; }
    case OP_RING_SYNC: z = sync_all(s); break;
    case OP_SYNC: z = cuStreamSynchronize(smap(s, a[0])); break;
    case OP_STREAM_QUERY: z = cuStreamQuery(smap(s, a[0])); break;
    case OP_STREAM_CREATE: { CUstream q = 0; z = cuStreamCreateWithPriority(&q, CU_STREAM_NON_BLOCKING, (int)(int64_t)a[1]);
      r[0] = (uint64_t)q; if (!z) track(s, RS_STREAM, r[0], 0, !(a[0] & CU_STREAM_NON_BLOCKING)); break; }
    case OP_STREAM_DESTROY: if (!find(s, RS_STREAM, a[0])) { z = CUDA_ERROR_INVALID_HANDLE; break; }
      cuStreamSynchronize((CUstream)a[0]); z = cuStreamDestroy((CUstream)a[0]); untrack(s, RS_STREAM, a[0]); break;
    case OP_STREAM_WAIT_EVENT: z = cuStreamWaitEvent(smap(s, a[0]), (CUevent)a[1], (unsigned)a[2]); break;
    case OP_EVENT_CREATE: { CUevent e = 0; z = cuEventCreate(&e, (unsigned)a[0]); r[0] = (uint64_t)e; if (!z) track(s, RS_EVENT, r[0], 0, 0); break; }
    case OP_EVENT_DESTROY: if (!find(s, RS_EVENT, a[0])) { z = CUDA_ERROR_INVALID_HANDLE; break; } z = cuEventDestroy((CUevent)a[0]); untrack(s, RS_EVENT, a[0]); break;
    case OP_EVENT_RECORD: prep_stream(s, a[1]); z = cuEventRecord((CUevent)a[0], smap(s, a[1])); break;
    case OP_EVENT_SYNC: z = cuEventSynchronize((CUevent)a[0]); break;
    case OP_EVENT_QUERY: z = cuEventQuery((CUevent)a[0]); break;
    case OP_EVENT_ELAPSED: { float ms = 0; z = cuEventElapsedTime(&ms, (CUevent)a[0], (CUevent)a[1]); memcpy(&r[0], &ms, 4); break; }
    case OP_CTX_SYNC: z = sync_all(s); break;
    case OP_CTX_LIMIT:
      if (a[2]) { size_t cur = 0; cuCtxGetLimit(&cur, (CUlimit)a[0]); z = a[1] > cur ? cuCtxSetLimit((CUlimit)a[0], a[1]) : 0; }  // shared: only ever raise
      else { size_t v = 0; z = cuCtxGetLimit(&v, (CUlimit)a[0]); r[0] = v; }
      break;
    // stream capture -> CUDA graphs. Capture is forced to RELAXED mode: "global" mode would forbid
    // other tenants' calls in other worker threads.
    case OP_CAPTURE_BEGIN: prep_stream(s, a[0]); z = cuStreamBeginCapture(smap(s, a[0]), CU_STREAM_CAPTURE_MODE_RELAXED); break;
    case OP_CAPTURE_END: { CUgraph g = 0; z = cuStreamEndCapture(smap(s, a[0]), &g); r[0] = (uint64_t)g; if (g) track(s, RS_GRAPH, r[0], 0, 0); break; }
    case OP_CAPTURE_INFO: { CUstreamCaptureStatus cs = 0; cuuint64_t id = 0; z = cuStreamGetCaptureInfo(smap(s, a[0]), &cs, &id, 0, 0, 0); r[0] = cs; r[1] = id; break; }
    case OP_GRAPH_INSTANTIATE: { CUgraphExec e = 0; z = cuGraphInstantiateWithFlags(&e, (CUgraph)a[0], a[1]); r[0] = (uint64_t)e; if (!z) track(s, RS_GEXEC, r[0], 0, 0); break; }
    case OP_GRAPH_LAUNCH: prep_stream(s, a[1]); z = cuGraphLaunch((CUgraphExec)a[0], smap(s, a[1])); break;
    case OP_GRAPH_EXEC_DESTROY: if (!untrack(s, RS_GEXEC, a[0])) { z = CUDA_ERROR_INVALID_HANDLE; break; } z = cuGraphExecDestroy((CUgraphExec)a[0]); break;
    case OP_GRAPH_DESTROY: if (!untrack(s, RS_GRAPH, a[0])) { z = CUDA_ERROR_INVALID_HANDLE; break; } z = cuGraphDestroy((CUgraph)a[0]); break;
    // JIT linking (cuLink*): options are dropped (they carry client pointers, e.g. log buffers)
    case OP_LINK_CREATE: { CUlinkState l = 0; z = cuLinkCreate(0, 0, 0, &l); r[0] = (uint64_t)l; if (!z) track(s, RS_LINK, r[0], 0, 0); break; }
    case OP_LINK_ADD: {   // a0 state, a1 type, a2 size, a3 from blob; data: [name 256][payload if inline]
      if (!find(s, RS_LINK, a[0])) { z = CUDA_ERROR_INVALID_HANDLE; break; }
      d[255] = 0; const void* src = a[3] ? (const void*)s->blob : (const void*)(d + 256);
      if (a[3] ? (!s->blob || s->blobn < a[2]) : !INRPC(256, a[2])) { z = CUDA_ERROR_INVALID_VALUE; break; }
      z = cuLinkAddData((CUlinkState)a[0], (CUjitInputType)a[1], (void*)src, a[2], (const char*)d, 0, 0, 0); break; }
    case OP_LINK_COMPLETE: {
      if (!find(s, RS_LINK, a[0])) { z = CUDA_ERROR_INVALID_HANDLE; break; }
      void* cub = 0; size_t n = 0; z = cuLinkComplete((CUlinkState)a[0], &cub, &n);
      if (!z) { if (n > rpc_size) z = CUDA_ERROR_INVALID_VALUE; else memcpy(d, cub, n); } r[0] = n; break; }
    case OP_LINK_DESTROY: if (!untrack(s, RS_LINK, a[0])) { z = CUDA_ERROR_INVALID_HANDLE; break; } z = cuLinkDestroy((CUlinkState)a[0]); break;
    case OP_GRAPH_NODES: {   // a0 graph, a1 capacity -> r0 count; data: node handles
      if (!find(s, RS_GRAPH, a[0])) { z = CUDA_ERROR_INVALID_HANDLE; break; }
      size_t n = 0; z = cuGraphGetNodes((CUgraph)a[0], 0, &n); r[0] = n;
      if (!z && a[1]) { size_t k = a[1] < n ? a[1] : n; if (!INRPC(0, k * 8)) { z = CUDA_ERROR_INVALID_VALUE; break; } z = cuGraphGetNodes((CUgraph)a[0], (CUgraphNode*)d, &k); r[0] = k; }
      break; }
    case OP_MEMCPY2D: {   // data: CUDA_MEMCPY2D; host-side rows (if any) packed at data+256
      CUDA_MEMCPY2D m; memcpy(&m, d, sizeof m);
      if (a[0] & 1) { m.srcMemoryType = CU_MEMORYTYPE_HOST; m.srcHost = d + 256; }
      if (a[0] & 2) { m.dstMemoryType = CU_MEMORYTYPE_HOST; m.dstHost = d + 256; }
      if ((a[0] & 3) && !INRPC(256, m.WidthInBytes * m.Height)) { z = CUDA_ERROR_INVALID_VALUE; break; }
      if (m.srcMemoryType == CU_MEMORYTYPE_ARRAY || m.dstMemoryType == CU_MEMORYTYPE_ARRAY) { z = CUDA_ERROR_NOT_SUPPORTED; break; }
      CUstream q = smap(s, a[1]); prep_stream(s, a[1]);
      z = cuMemcpy2DAsync(&m, q); if (!z && (a[0] & 3)) z = cuStreamSynchronize(q); break; }
    default: z = CUDA_ERROR_NOT_SUPPORTED;
  }
  (void)idx; return z;
}

static void* serve(void* arg) {
  (void)arg; cuCtxSetCurrent(ctx);
  for (;;) {
    struct epoll_event ev; int n = epoll_wait(epfd, &ev, 1, -1); if (n <= 0) continue;
    int i = (int)ev.data.u32; sess_t* s = &S[i]; uint64_t v;
    while (read(s->efd, &v, 8) == 8) {}
    pthread_mutex_lock(&s->mu);
    wd_page_t* p = s->page;
    if (p->state == WD_REQ) {
      if (!s->open) { struct timespec t; clock_gettime(CLOCK_REALTIME, &t); t.tv_sec += 2;
        while (!s->open && pthread_cond_timedwait(&s->cv, &s->mu, &t) == 0) {} }
      double t0 = wd_now_us(); int op = p->op;
      CUresult z = s->open ? dispatch(i, s) : CUDA_ERROR_NOT_INITIALIZED;
      if (op > 0 && op < OP_MAX) ST->op_count[op]++;
      ST->rpc_total++; ST->rpc_ns += (uint64_t)((wd_now_us() - t0) * 1e3);
      p->err = (int)z; __sync_synchronize(); p->state = WD_DONE; wd_futex_wake(&p->state);
      if (is_sticky(z)) {
        ST->faults++;
        LOGE("sticky CUDA error %d in session %u (op %d): context is dead, exiting for failover", z, s->sid, op);
        wd_ctl_t m = {C_FAULT, (uint32_t)i, {(uint64_t)z, s->sid, 0, 0}}; wd_send_fds(ctl, &m, sizeof m, 0, 0);
        _exit(3);
      }
    }
    pthread_mutex_unlock(&s->mu);
    struct epoll_event re = {EPOLLIN | EPOLLONESHOT, {.u32 = (uint32_t)i}}; epoll_ctl(epfd, EPOLL_CTL_MOD, s->efd, &re);
  }
  return 0;
}

int worker_main(int argc, char** argv) {
  extern const char* wd_log_comp; wd_log_comp = "worker";
  extern int wd_log_level;
  for (int i = 2; i + 1 < argc; i += 2) {
    if (!strcmp(argv[i], "--ctl-fd")) ctl = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--gpu")) gpu = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--threads")) nthreads = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--log-level")) wd_log_level = atoi(argv[i + 1]);
  }
  sigset_t ss; sigemptyset(&ss); sigprocmask(SIG_SETMASK, &ss, 0); signal(SIGPIPE, SIG_IGN);
  if (ctl < 0) { LOGE("no --ctl-fd"); return 2; }
  ST = &st_dummy;
  double t0 = wd_now_us();
  CKX(cuInit(0)); CKX(cuDeviceGet(&dev, gpu)); CKX(cuDevicePrimaryCtxRetain(&ctx, dev)); CKX(cuCtxSetCurrent(ctx));
  epfd = epoll_create1(EPOLL_CLOEXEC);
  int promoted = 0; pthread_t th[64];
  for (;;) {
    wd_ctl_t m; int fds[4], nfd = 0;
    ssize_t n = wd_recv_fds(ctl, &m, sizeof m, fds, &nfd);
    if (n <= 0) { LOGW("supervisor gone; exiting"); _exit(0); }
    switch (m.type) {
      case C_ATTACH:
        if (m.region == UINT32_MAX) {     // stats page: last attach message -> report READY
          ST = mmap(0, sizeof(wd_stats_t), PROT_READ | PROT_WRITE, MAP_SHARED, fds[0], 0); close(fds[0]);
          if (ST == MAP_FAILED) ST = &st_dummy;
          size_t fr, tot; cuMemGetInfo(&fr, &tot); ST->gpu_mem_free = fr; ST->gpu_mem_total = tot;
          wd_ctl_t rd = {C_READY, 0, {(uint64_t)(wd_now_us() - t0), tot - fr, 0, 0}}; wd_send_fds(ctl, &rd, sizeof rd, 0, 0);
        } else if (m.region < WD_MAX_SESSIONS && nfd == 2) {
          sess_t* s = &S[m.region]; size_t sz = m.arg[0];
          s->page = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fds[0], 0); close(fds[0]);
          if (s->page == MAP_FAILED) { LOGE("mmap region %u: %s", m.region, strerror(errno)); _exit(2); }
          s->data = WD_DATA(s->page); s->efd = fds[1]; s->size = sz;
          rpc_size = m.arg[1]; ring_off = rpc_size; ring_size = sz - WD_PAGE - rpc_size;
          CKX(cuMemHostRegister(s->page, sz, CU_MEMHOSTREGISTER_PORTABLE));
          CKX(cuStreamCreate(&s->s0, CU_STREAM_NON_BLOCKING));
          pthread_mutex_init(&s->mu, 0); pthread_cond_init(&s->cv, 0); s->attached = 1;
          if ((int)m.region >= nsess) nsess = m.region + 1;
          struct epoll_event e = {EPOLLIN | EPOLLONESHOT, {.u32 = m.region}}; epoll_ctl(epfd, EPOLL_CTL_ADD, s->efd, &e);
        }
        break;
      case C_OPEN: { sess_t* s = &S[m.region]; pthread_mutex_lock(&s->mu);
        if (s->open) close_session(s);
        s->open = 1; s->sid = (uint32_t)m.arg[0]; s->quota = m.arg[1]; s->used = 0;
        pthread_cond_broadcast(&s->cv); pthread_mutex_unlock(&s->mu); break; }
      case C_CLOSE: { sess_t* s = &S[m.region]; pthread_mutex_lock(&s->mu); close_session(s); pthread_mutex_unlock(&s->mu);
        wd_ctl_t c = {C_CLOSED, m.region, {m.arg[0], 0, 0, 0}}; wd_send_fds(ctl, &c, sizeof c, 0, 0); break; }
      case C_PROMOTE:
        if (!promoted) { promoted = 1; for (int t = 0; t < nthreads && t < 64; t++) pthread_create(&th[t], 0, serve, 0); }
        { wd_ctl_t c = {C_ACTIVE, 0, {0}}; wd_send_fds(ctl, &c, sizeof c, 0, 0); }
        LOGI("ACTIVE with %d dispatch threads", nthreads); break;
      case C_STOP: LOGI("stop"); _exit(0);
    }
  }
}
