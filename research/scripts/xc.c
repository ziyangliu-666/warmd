// H19: share a device-memory segment across container boundaries. The owner and the client run in
// any two containers that share only a directory holding the socket. Native CUDA with no torch.
//   xc owner <sock> <mib> [exit]   create, fill, export, and serve the fd; "exit" leaves after one send
//   xc client <sock> <mib>         import, map read-only, read, then try to write
#define _GNU_SOURCE
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>
#define CK(x) do { CUresult r_ = (x); if (r_) { const char* s_ = ""; cuGetErrorString(r_, &s_); printf("  FAIL %s: %d %s\n", #x, r_, s_); fflush(stdout); return 1; } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
// The primary context API has one version on every driver branch; cuCtxCreate does not.
static int open_ctx(CUdevice* dev) { CUcontext c; CK(cuInit(0)); CK(cuDeviceGet(dev, 0)); CK(cuDevicePrimaryCtxRetain(&c, *dev)); CK(cuCtxSetCurrent(c)); return 0; }
static int map(CUmemGenericAllocationHandle h, size_t n, CUdevice dev, int flag, CUdeviceptr* va) {
  CK(cuMemAddressReserve(va, n, 0, 0, 0)); CK(cuMemMap(*va, n, 0, h, 0));
  CUmemAccessDesc d; memset(&d, 0, sizeof d); d.location.type = CU_MEM_LOCATION_TYPE_DEVICE; d.location.id = dev; d.flags = flag; CK(cuMemSetAccess(*va, n, &d, 1)); return 0; }
static int owner(const char* path, size_t n, int leave) {
  CUdevice dev; if (open_ctx(&dev)) return 1;
  CUmemAllocationProp p; memset(&p, 0, sizeof p); p.type = CU_MEM_ALLOCATION_TYPE_PINNED; p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR; p.location.type = CU_MEM_LOCATION_TYPE_DEVICE; p.location.id = dev;
  CUmemGenericAllocationHandle h; CK(cuMemCreate(&h, n, &p, 0)); CUdeviceptr va; if (map(h, n, dev, CU_MEM_ACCESS_FLAGS_PROT_READWRITE, &va)) return 1;
  CK(cuMemsetD32(va, 0xabcd1234, n / 4)); CK(cuCtxSynchronize()); int fd; CK(cuMemExportToShareableHandle(&fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
  int s = socket(AF_UNIX, SOCK_STREAM, 0); struct sockaddr_un a = {AF_UNIX}; strcpy(a.sun_path, path); unlink(path); if (bind(s, (void*)&a, sizeof a) || listen(s, 8)) { perror("bind"); return 1; } chmod(path, 0666);
  printf("  owner ready pid=%d uid=%d\n", getpid(), getuid()); fflush(stdout);
  for (;;) { int f = accept(s, 0, 0); char b = 'f'; struct iovec io = {&b, 1}; char cb[CMSG_SPACE(4)]; struct msghdr m = {.msg_iov = &io, .msg_iovlen = 1, .msg_control = cb, .msg_controllen = sizeof cb};
    struct cmsghdr* c = CMSG_FIRSTHDR(&m); c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS; c->cmsg_len = CMSG_LEN(4); memcpy(CMSG_DATA(c), &fd, 4); sendmsg(f, &m, 0); close(f); if (leave) { CK(cuMemUnmap(va, n)); CK(cuMemRelease(h)); return 0; } }
}
static int client(const char* path, size_t n) {
  int s = socket(AF_UNIX, SOCK_STREAM, 0); struct sockaddr_un a = {AF_UNIX}; strcpy(a.sun_path, path); if (connect(s, (void*)&a, sizeof a)) { perror("  FAIL connect"); return 1; } printf("  connect ok pid=%d uid=%d\n", getpid(), getuid());
  char b; struct iovec io = {&b, 1}; char cb[CMSG_SPACE(4)]; struct msghdr m = {.msg_iov = &io, .msg_iovlen = 1, .msg_control = cb, .msg_controllen = sizeof cb}; if (recvmsg(s, &m, 0) != 1 || !CMSG_FIRSTHDR(&m)) { puts("  FAIL recvmsg, no fd"); return 1; }
  int fd; memcpy(&fd, CMSG_DATA(CMSG_FIRSTHDR(&m)), 4); char l[128] = ""; char q[64]; snprintf(q, 64, "/proc/self/fd/%d", fd); readlink(q, l, 127); printf("  fd ok -> %s\n", l); fflush(stdout);
  CUdevice dev; if (open_ctx(&dev)) return 1; size_t f0, t; CK(cuMemGetInfo(&f0, &t));
  double t0 = now(); CUmemGenericAllocationHandle h; CK(cuMemImportFromShareableHandle(&h, (void*)(uintptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)); double imp = now() - t0;
  CUdeviceptr va; t0 = now(); if (map(h, n, dev, CU_MEM_ACCESS_FLAGS_PROT_READ, &va)) return 1; double mp = now() - t0; size_t f1; CK(cuMemGetInfo(&f1, &t));
  uint32_t x = 0, y = 0, z = 0; CK(cuMemcpyDtoH(&x, va, 4)); CK(cuMemcpyDtoH(&y, va + n - 4, 4)); CUresult w = cuMemcpyHtoD(va, &z, 4);
  printf("  PASS import_ms=%.2f map_ms=%.2f delta_mib=%.0f head=%x tail=%x write_result=%d\n", imp * 1e3, mp * 1e3, (double)(long)(f0 - f1) / 1048576.0, x, y, w); return x == 0xabcd1234 && y == 0xabcd1234 ? 0 : 2;
}
int main(int c, char** v) { size_t n = (size_t)atol(v[3]) << 20; return !strcmp(v[1], "owner") ? owner(v[2], n, c > 4) : client(v[2], n); }
