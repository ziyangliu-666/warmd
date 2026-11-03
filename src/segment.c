// segment.c: thin wrapper over the public CUDA virtual memory management API, for warmd.segments.
//
// Python calls these through ctypes with scalar arguments only, so no CUDA struct layout is
// defined twice. Every function acts on the calling thread's current context and returns a
// CUresult. See docs/segments.md.
#include <cuda.h>
#include <stdint.h>
#include <string.h>

static int current_device(CUdevice* d) { return cuCtxGetDevice(d); }

// Physical device memory that can be exported as a POSIX file descriptor.
int seg_create(size_t n, uint64_t* handle) {
  CUdevice d; int r = current_device(&d); if (r) return r;
  CUmemAllocationProp p; memset(&p, 0, sizeof p);
  p.type = CU_MEM_ALLOCATION_TYPE_PINNED; p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  p.location.type = CU_MEM_LOCATION_TYPE_DEVICE; p.location.id = d;
  return cuMemCreate((CUmemGenericAllocationHandle*)handle, n, &p, 0);
}

int seg_granularity(size_t* g) {
  CUdevice d; int r = current_device(&d); if (r) return r;
  CUmemAllocationProp p; memset(&p, 0, sizeof p);
  p.type = CU_MEM_ALLOCATION_TYPE_PINNED; p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  p.location.type = CU_MEM_LOCATION_TYPE_DEVICE; p.location.id = d;
  return cuMemGetAllocationGranularity(g, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
}

int seg_export(uint64_t handle, int* fd) {
  return cuMemExportToShareableHandle(fd, (CUmemGenericAllocationHandle)handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
}

int seg_import(int fd, uint64_t* handle) {
  return cuMemImportFromShareableHandle((CUmemGenericAllocationHandle*)handle, (void*)(uintptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
}

int seg_release(uint64_t handle) { return cuMemRelease((CUmemGenericAllocationHandle)handle); }

// A virtual range that a client keeps for its lifetime. Tensors point into it, so they stay
// valid while the physical memory behind it is unmapped, evicted and mapped again.
int seg_reserve(size_t n, size_t align, uint64_t* va) { return cuMemAddressReserve((CUdeviceptr*)va, n, align, 0, 0); }
int seg_unreserve(uint64_t va, size_t n) { return cuMemAddressFree((CUdeviceptr)va, n); }

// read_only != 0 maps with CU_MEM_ACCESS_FLAGS_PROT_READ: a kernel store faults the writer only.
int seg_map(uint64_t va, size_t n, uint64_t handle, int read_only) {
  CUdevice d; int r = current_device(&d); if (r) return r;
  r = cuMemMap((CUdeviceptr)va, n, 0, (CUmemGenericAllocationHandle)handle, 0); if (r) return r;
  CUmemAccessDesc a; memset(&a, 0, sizeof a);
  a.location.type = CU_MEM_LOCATION_TYPE_DEVICE; a.location.id = d;
  a.flags = read_only ? CU_MEM_ACCESS_FLAGS_PROT_READ : CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  r = cuMemSetAccess((CUdeviceptr)va, n, &a, 1);
  if (r) cuMemUnmap((CUdeviceptr)va, n);
  return r;
}

int seg_unmap(uint64_t va, size_t n) { return cuMemUnmap((CUdeviceptr)va, n); }

int seg_write(uint64_t va, const void* src, size_t n) { return cuMemcpyHtoD((CUdeviceptr)va, src, n); }

// Pinned host memory. A copy from it runs at full PCIe speed.
int seg_host_alloc(void** p, size_t n) { return cuMemHostAlloc(p, n, 0); }
int seg_host_free(void* p) { return cuMemFreeHost(p); }
