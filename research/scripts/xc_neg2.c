// Negative checks for version 2 of the nvproxy change: only the NVIDIA control device may import.
#define _GNU_SOURCE
#include <cuda.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
int main(void) {
  CUdevice d; CUcontext c; cuInit(0); cuDeviceGet(&d, 0); cuDevicePrimaryCtxRetain(&c, d); cuCtxSetCurrent(c);
  const char* f[] = {"/x/xc", "/dev/null", "/dev/nvidia0", "/dev/nvidiactl"}; int bad = 0;
  for (int i = 0; i < 4; i++) {
    int fd = open(f[i], O_RDONLY); CUmemGenericAllocationHandle h = 0;
    CUresult r = cuMemImportFromShareableHandle(&h, (void*)(uintptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    printf("  %-16s result=%d %s\n", f[i], r, r ? "rejected" : "ACCEPTED"); bad += r == 0;
  }
  return bad;
}
