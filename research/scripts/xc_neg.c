// Negative check for the nvproxy patch: an ordinary host file must not import as device memory.
#define _GNU_SOURCE
#include <cuda.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
int main(void) {
  CUdevice d; CUcontext c; cuInit(0); cuDeviceGet(&d, 0); cuDevicePrimaryCtxRetain(&c, d); cuCtxSetCurrent(c);
  int fd = open("/x/xc", O_RDONLY); CUmemGenericAllocationHandle h = 0;
  CUresult r = cuMemImportFromShareableHandle(&h, (void*)(uintptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  printf("  import of an ordinary file: result=%d (%s)\n", r, r ? "rejected" : "ACCEPTED"); return r == 0;
}
