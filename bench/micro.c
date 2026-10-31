// micro.c - driver-API micro benchmarks. Build twice: against the warmd shim and against the real
// libcuda, to compare. Prints JSON lines.
//   micro rpc      : round trip of a trivial GPU call (cuCtxSynchronize), and empty kernel launch+sync
//   micro copy     : H2D / D2H bandwidth, pageable (sync) and pinned (async + sync), 4 KB .. 256 MB
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
#define CK(x) do { CUresult r_ = (x); if (r_) { const char* s; cuGetErrorString(r_, &s); fprintf(stderr, "%s -> %s\n", #x, s); exit(1); } } while (0)
static const char PTX[] = ".version 7.0\n.target sm_80\n.address_size 64\n.visible .entry nop()\n{\n ret;\n}\n";
static int cmpd(const void* a, const void* b) { double x = *(double*)a, y = *(double*)b; return x < y ? -1 : x > y; }
static void pct(const char* what, double* v, int n, const char* label) {
  qsort(v, n, sizeof *v, cmpd);
  printf("{\"bench\":\"%s\",\"impl\":\"%s\",\"n\":%d,\"p50_us\":%.1f,\"p99_us\":%.1f,\"min_us\":%.1f}\n", what, label, n, v[n / 2], v[n * 99 / 100], v[0]);
}
int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "rpc"; const char* label = getenv("IMPL") ? getenv("IMPL") : "?";
  double t0 = us(); CUdevice d; CUcontext c; CK(cuInit(0)); CK(cuDeviceGet(&d, 0)); CK(cuDevicePrimaryCtxRetain(&c, d)); CK(cuCtxSetCurrent(c));
  double tinit = us() - t0; CUstream s; CK(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));
  if (!strcmp(mode, "rpc")) {
    enum { N = 5000 }; static double v[N];
    printf("{\"bench\":\"init\",\"impl\":\"%s\",\"ms\":%.2f}\n", label, tinit / 1e3);
    for (int i = 0; i < N; i++) { double a = us(); CK(cuStreamSynchronize(s)); v[i] = us() - a; } pct("stream_sync_roundtrip", v, N, label);
    CUmodule m; CUfunction f; CK(cuModuleLoadData(&m, PTX)); CK(cuModuleGetFunction(&f, m, "nop"));
    for (int i = 0; i < N; i++) { double a = us(); CK(cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, s, 0, 0)); CK(cuStreamSynchronize(s)); v[i] = us() - a; } pct("launch_sync", v, N, label);
    double a = us(); for (int i = 0; i < N; i++) CK(cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, s, 0, 0)); CK(cuStreamSynchronize(s));
    printf("{\"bench\":\"launch_throughput\",\"impl\":\"%s\",\"launches_per_s\":%.0f}\n", label, N / ((us() - a) / 1e6));
  } else {
    size_t maxn = 256u << 20; CUdeviceptr dp; CK(cuMemAlloc(&dp, maxn));
    char* pg = malloc(maxn); memset(pg, 1, maxn); void* pin; CK(cuMemHostAlloc(&pin, maxn, 0)); memset(pin, 1, maxn);
    for (size_t n = 4096; n <= maxn; n *= 8) {
      int it = n < (1 << 20) ? 200 : n < (32u << 20) ? 20 : 5; double a;
      a = us(); for (int i = 0; i < it; i++) CK(cuMemcpyHtoD(dp, pg, n)); double h2d = (us() - a) / it;
      a = us(); for (int i = 0; i < it; i++) CK(cuMemcpyDtoH(pg, dp, n)); double d2h = (us() - a) / it;
      a = us(); for (int i = 0; i < it; i++) CK(cuMemcpyHtoDAsync(dp, pin, n, s)); CK(cuStreamSynchronize(s)); double h2dp = (us() - a) / it;
      a = us(); for (int i = 0; i < it; i++) CK(cuMemcpyDtoHAsync(pin, dp, n, s)); CK(cuStreamSynchronize(s)); double d2hp = (us() - a) / it;
      printf("{\"bench\":\"copy\",\"impl\":\"%s\",\"bytes\":%zu,\"h2d_pageable_GBs\":%.2f,\"d2h_pageable_GBs\":%.2f,\"h2d_pinned_async_GBs\":%.2f,\"d2h_pinned_async_GBs\":%.2f,"
             "\"h2d_pageable_us\":%.1f,\"d2h_pageable_us\":%.1f}\n", label, n, n / h2d / 1e3, n / d2h / 1e3, n / h2dp / 1e3, n / d2hp / 1e3, h2d, d2h);
    }
  }
  return 0;
}
