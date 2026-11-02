// vf: thin wrapper over the public CUDA VMM API, called from Python with scalar arguments only.
// Every function acts on the calling thread's current context. Returns a CUresult.
#include <cuda.h>
#include <string.h>
#include <stdint.h>
static CUdevice dev0(void) { CUdevice d; cuCtxGetDevice(&d); return d; }
int vf_create(size_t n, uint64_t* h) {
  CUmemAllocationProp p; memset(&p, 0, sizeof p); p.type = CU_MEM_ALLOCATION_TYPE_PINNED; p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  p.location.type = CU_MEM_LOCATION_TYPE_DEVICE; p.location.id = dev0();
  return cuMemCreate((CUmemGenericAllocationHandle*)h, n, &p, 0);
}
int vf_export(uint64_t h, int* fd) { return cuMemExportToShareableHandle(fd, (CUmemGenericAllocationHandle)h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0); }
int vf_import(int fd, uint64_t* h) { return cuMemImportFromShareableHandle((CUmemGenericAllocationHandle*)h, (void*)(uintptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR); }
int vf_reserve(size_t n, uint64_t* va) { return cuMemAddressReserve((CUdeviceptr*)va, n, 2u << 20, 0, 0); }
int vf_map(uint64_t va, size_t n, uint64_t h, int ro) {
  int r = cuMemMap((CUdeviceptr)va, n, 0, (CUmemGenericAllocationHandle)h, 0); if (r) return r;
  CUmemAccessDesc d; memset(&d, 0, sizeof d); d.location.type = CU_MEM_LOCATION_TYPE_DEVICE; d.location.id = dev0();
  d.flags = ro ? CU_MEM_ACCESS_FLAGS_PROT_READ : CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  return cuMemSetAccess((CUdeviceptr)va, n, &d, 1);
}
int vf_unmap(uint64_t va, size_t n) { return cuMemUnmap((CUdeviceptr)va, n); }
int vf_release(uint64_t h) { return cuMemRelease((CUmemGenericAllocationHandle)h); }
int vf_write(uint64_t va, const void* src, size_t n) { return cuMemcpyHtoD((CUdeviceptr)va, src, n); }
int vf_pin(void** p, size_t n) { return cuMemHostAlloc(p, n, 0); }
