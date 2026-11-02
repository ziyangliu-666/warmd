// H5: split warmd's host-to-device path into its parts. Native CUDA, no RPC.
// A shared region is registered, data is memcpy'd in, then DMA'd in chunks, as in h2d_sync.
#define _GNU_SOURCE
#include <cuda.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#define CK(x) do { CUresult r_ = (x); if (r_) { printf("error %d at %s\n", r_, #x); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
typedef struct { char* d; const char* s; size_t n; } job;
static void* mc(void* a) { job* j = a; memcpy(j->d, j->s, j->n); return 0; }
static void pcopy(char* d, const char* s, size_t n, int th) {
  pthread_t t[16]; job j[16]; size_t c = n / th;
  for (int i = 0; i < th; i++) { j[i] = (job){d + i * c, s + i * c, i == th - 1 ? n - c * i : c}; pthread_create(&t[i], 0, mc, &j[i]); }
  for (int i = 0; i < th; i++) pthread_join(t[i], 0);
}
int main(void) {
  CK(cuInit(0)); CUdevice dev; CUcontext ctx; CK(cuDeviceGet(&dev, 0)); CK(cuCtxCreate(&ctx, 0, dev));
  size_t total = 2ul << 30; char* src = aligned_alloc(4096, total); memset(src, 1, total);
  CUdeviceptr dst; CK(cuMemAlloc(&dst, total)); CUstream q; CK(cuStreamCreate(&q, 0));
  { double t = now(); CK(cuMemcpyHtoD(dst, src, total)); printf("{\"path\":\"native_pageable\",\"gbs\":%.1f}\n", total / 1e9 / (now() - t)); }
  { void* p; CK(cuMemHostAlloc(&p, total, 0)); memset(p, 1, total); double t = now(); CK(cuMemcpyHtoD(dst, p, total));
    printf("{\"path\":\"native_pinned\",\"gbs\":%.1f}\n", total / 1e9 / (now() - t));
    t = now(); CK(cuMemcpyHtoD(dst, p, total)); printf("{\"path\":\"native_pinned_second\",\"gbs\":%.1f}\n", total / 1e9 / (now() - t)); CK(cuMemFreeHost(p)); }
  { void* r = mmap(0, total, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0); memset(r, 1, total); CK(cuMemHostRegister(r, total, CU_MEMHOSTREGISTER_PORTABLE));
    double t = now(); CK(cuMemcpyHtoD(dst, r, total)); printf("{\"path\":\"registered_shared_zero_copy\",\"gbs\":%.1f}\n", total / 1e9 / (now() - t)); }
  size_t sizes[] = {8ul << 20, 64ul << 20, 256ul << 20}; int ths[] = {1, 4, 8};
  for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) {
    size_t sz = sizes[a]; int th = ths[b];
    char* r = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0); memset(r, 0, sz); CK(cuMemHostRegister(r, sz, CU_MEMHOSTREGISTER_PORTABLE));
    double cp = 0, dm = 0, t0 = now();
    for (size_t o = 0; o < total; o += sz) { size_t c = total - o < sz ? total - o : sz; double t = now(); pcopy(r, src + o, c, th); cp += now() - t;
      t = now(); CK(cuMemcpyHtoDAsync(dst + o, r, c, q)); CK(cuStreamSynchronize(q)); dm += now() - t; }
    printf("{\"path\":\"staged\",\"staging_mb\":%zu,\"threads\":%d,\"gbs\":%.1f,\"memcpy_gbs\":%.1f,\"dma_gbs\":%.1f}\n", sz >> 20, th, total / 1e9 / (now() - t0), total / 1e9 / cp, total / 1e9 / dm);
    CK(cuMemHostUnregister(r)); munmap(r, sz);
  }
  return 0;
}
