// H28: what is in a context? Prints device memory and time for each step of a bare native start.
// Environment variables select driver behaviour. No PyTorch.
#define _GNU_SOURCE
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(void) {
  double t = now(); if (cuInit(0)) return 1; double init = now() - t; CUdevice d; cuDeviceGet(&d, 0); CUcontext c;
  t = now(); if (cuDevicePrimaryCtxRetain(&c, d)) return 2; cuCtxSetCurrent(c); double ctx = now() - t;
  size_t f, tot; cuMemGetInfo(&f, &tot); t = now(); CUdeviceptr p; cuMemAlloc(&p, 4096); double al = now() - t; size_t f2; cuMemGetInfo(&f2, &tot);
  printf("{\"init_ms\":%.1f,\"ctx_ms\":%.1f,\"first_alloc_ms\":%.2f,\"used_by_ctx_mib\":%.0f,\"used_after_alloc_mib\":%.0f}\n", init * 1e3, ctx * 1e3, al * 1e3, (double)(tot - f) / 1048576, (double)(tot - f2) / 1048576); fflush(stdout);
  if (getenv("HOLD")) sleep(atoi(getenv("HOLD"))); return 0;
}
