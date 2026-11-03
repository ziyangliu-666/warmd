// H6: a device-memory file. An owner process creates physical device memory and exports it as a
// POSIX fd. Other native processes, each with its own CUDA context, import it and map it read-only
// at a virtual address they choose. No driver interposition and no shared context.
//
//   h6_vramfile <size_gb>
//
// Checks, each reported separately:
//   export/import cost and device memory delta (zero copy means the delta is about one context)
//   reader sees the owner's bytes
//   a write through the read-only mapping faults the writer only
//   the owner and other readers keep working after that fault
//   a reader's exit does not free the memory
#define _GNU_SOURCE
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CK(x) do { CUresult r_ = (x); if (r_) { const char* s_ = ""; cuGetErrorString(r_, &s_); printf("{\"error\":\"%s\",\"at\":\"%s\"}\n", s_, #x); fflush(stdout); _exit(1); } } while (0)

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void send_fd(int s, int fd) {
  char b = 'f'; struct iovec io = {&b, 1}; char cb[CMSG_SPACE(4)]; struct msghdr h = {.msg_iov = &io, .msg_iovlen = 1, .msg_control = cb, .msg_controllen = sizeof cb};
  struct cmsghdr* c = CMSG_FIRSTHDR(&h); c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS; c->cmsg_len = CMSG_LEN(4); memcpy(CMSG_DATA(c), &fd, 4);
  if (sendmsg(s, &h, 0) != 1) { perror("sendmsg"); _exit(1); }
}
static int recv_fd(int s) {
  char b; struct iovec io = {&b, 1}; char cb[CMSG_SPACE(4)]; struct msghdr h = {.msg_iov = &io, .msg_iovlen = 1, .msg_control = cb, .msg_controllen = sizeof cb};
  if (recvmsg(s, &h, 0) != 1) { perror("recvmsg"); _exit(1); }
  int fd; memcpy(&fd, CMSG_DATA(CMSG_FIRSTHDR(&h)), 4); return fd;
}
static size_t vram_used(CUdevice d) { size_t f, t; CK(cuMemGetInfo(&f, &t)); (void)d; return t - f; }

// Maps the imported file. flags selects read-only or read-write access.
static CUdeviceptr map_file(CUmemGenericAllocationHandle h, size_t n, CUdevice dev, CUmemAccess_flags flags, double* ms) {
  double t = now(); CUdeviceptr va; CK(cuMemAddressReserve(&va, n, 0, 0, 0)); CK(cuMemMap(va, n, 0, h, 0));
  CUmemAccessDesc d; memset(&d, 0, sizeof d); d.location.type = CU_MEM_LOCATION_TYPE_DEVICE; d.location.id = dev; d.flags = flags;
  CK(cuMemSetAccess(va, n, &d, 1)); *ms = (now() - t) * 1e3; return va;
}

static void reader(int s, size_t n, int mode) {
  // mode 0: read and report, then idle until told to recheck. mode 1: try to write through the mapping.
  CK(cuInit(0)); CUdevice dev; CUcontext ctx; CK(cuDeviceGet(&dev, 0)); CK(cuCtxCreate(&ctx, 0, dev));
  size_t before = vram_used(dev); int fd = recv_fd(s);
  double t = now(); CUmemGenericAllocationHandle h; CK(cuMemImportFromShareableHandle(&h, (void*)(uintptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)); double imp = (now() - t) * 1e3;
  double mp; CUdeviceptr va = map_file(h, n, dev, CU_MEM_ACCESS_FLAGS_PROT_READ, &mp);
  uint32_t head = 0, tail = 0, zero = 0; CUdeviceptr scratch; CK(cuMemAlloc(&scratch, 4096));
  t = now(); CK(cuMemcpyDtoH(&head, va, 4)); CK(cuMemcpyDtoH(&tail, va + n - 4, 4)); double rd = (now() - t) * 1e3;
  printf("{\"role\":\"reader\",\"mode\":%d,\"import_ms\":%.2f,\"map_ms\":%.2f,\"read_ms\":%.2f,\"head\":\"%x\",\"tail\":\"%x\",\"delta_mib\":%.0f}\n", mode, imp, mp, rd, head, tail, (vram_used(dev) - before) / 1048576.0); fflush(stdout);
  if (mode == 1) {
    CUresult r = cuMemcpyHtoD(va, &zero, 4); CUresult r2 = cuCtxSynchronize();
    uint32_t again = 0; CUresult r3 = cuMemcpyDtoH(&again, scratch, 4);
    printf("{\"role\":\"writer\",\"copy_write_result\":%d,\"sync_result\":%d,\"later_call_result\":%d}\n", r, r2, r3); fflush(stdout);
  } else {
    char b; if (read(s, &b, 1) != 1) _exit(1);
    uint32_t h2 = 0; CUresult r = cuMemcpyDtoH(&h2, va, 4); CK(cuCtxSynchronize());
    printf("{\"role\":\"reader_after_fault\",\"read_result\":%d,\"head\":\"%x\"}\n", r, h2); fflush(stdout);
  }
  _exit(0);
}

int main(int c, char** v) {
  size_t n = (size_t)(c > 1 ? atol(v[1]) : 8) << 30; int sp[2], sw[2];
  socketpair(AF_UNIX, SOCK_STREAM, 0, sp); socketpair(AF_UNIX, SOCK_STREAM, 0, sw);
  // Fork before any CUDA call in the owner: a native context does not survive fork().
  pid_t pr = fork(); if (!pr) reader(sp[1], n, 0);
  pid_t pw = fork(); if (!pw) reader(sw[1], n, 1);
  CK(cuInit(0)); CUdevice dev; CUcontext ctx; CK(cuDeviceGet(&dev, 0)); CK(cuCtxCreate(&ctx, 0, dev));
  size_t before = vram_used(dev);
  CUmemAllocationProp p; memset(&p, 0, sizeof p); p.type = CU_MEM_ALLOCATION_TYPE_PINNED; p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  p.location.type = CU_MEM_LOCATION_TYPE_DEVICE; p.location.id = dev;
  size_t gran; CK(cuMemGetAllocationGranularity(&gran, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  double t = now(); CUmemGenericAllocationHandle h; CK(cuMemCreate(&h, n, &p, 0)); double create = (now() - t) * 1e3;
  double mp; CUdeviceptr va = map_file(h, n, dev, CU_MEM_ACCESS_FLAGS_PROT_READWRITE, &mp);
  CK(cuMemsetD32(va, 0xabcd1234, n / 4)); CK(cuCtxSynchronize());
  int fd; t = now(); CK(cuMemExportToShareableHandle(&fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0)); double ex = (now() - t) * 1e3;
  printf("{\"role\":\"owner\",\"size_gb\":%zu,\"granularity_kib\":%zu,\"create_ms\":%.2f,\"map_ms\":%.2f,\"export_ms\":%.3f,\"delta_mib\":%.0f}\n", n >> 30, gran >> 10, create, mp, ex, (vram_used(dev) - before) / 1048576.0); fflush(stdout);
  send_fd(sp[0], fd); sleep(3);
  send_fd(sw[0], fd); int st; waitpid(pw, &st, 0);
  printf("{\"role\":\"writer_exit\",\"status\":%d}\n", WEXITSTATUS(st)); fflush(stdout);
  uint32_t x = 0; CUresult r = cuMemcpyDtoH(&x, va, 4);
  printf("{\"role\":\"owner_after_fault\",\"read_result\":%d,\"head\":\"%x\"}\n", r, x); fflush(stdout);
  char b = 'g'; if (write(sp[0], &b, 1) != 1) return 1; waitpid(pr, &st, 0);
  x = 0; r = cuMemcpyDtoH(&x, va + n - 4, 4);
  printf("{\"role\":\"owner_after_reader_exit\",\"read_result\":%d,\"tail\":\"%x\"}\n", r, x); fflush(stdout);
  return 0;
}
