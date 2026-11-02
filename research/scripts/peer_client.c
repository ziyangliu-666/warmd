// H2 client: runs inside the sandbox. Reports each primitive separately.
#define _GNU_SOURCE
#include <errno.h>
#include <linux/futex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int c, char** v) {
  int s = socket(AF_UNIX, SOCK_STREAM, 0); struct sockaddr_un a = {AF_UNIX}; strcpy(a.sun_path, v[1]);
  if (connect(s, (void*)&a, sizeof a)) { printf("connect: FAIL %s\n", strerror(errno)); return 1; } puts("connect: ok");
  char b; struct iovec io = {&b, 1}; char cb[CMSG_SPACE(8)]; struct msghdr h = {.msg_iov = &io, .msg_iovlen = 1, .msg_control = cb, .msg_controllen = sizeof cb};
  int n = recvmsg(s, &h, 0); struct cmsghdr* cm = CMSG_FIRSTHDR(&h);
  if (n != 1 || !cm || cm->cmsg_type != SCM_RIGHTS) { printf("scm_rights: FAIL n=%d cm=%p %s\n", n, (void*)cm, strerror(errno)); return 1; } int fds[2]; memcpy(fds, CMSG_DATA(cm), 8); puts("scm_rights: ok");
  volatile int32_t* w = mmap(0, 1 << 20, PROT_READ | PROT_WRITE, MAP_SHARED, fds[0], 0);
  if (w == MAP_FAILED) { printf("mmap: FAIL %s\n", strerror(errno)); return 1; } puts("mmap: ok");
  int N = 2000; double t = now(); int spin_ok = 0;
  for (int i = 0; i < N; i++) { w[0] = 1; uint64_t x = 1; if (write(fds[1], &x, 8) != 8) { printf("eventfd: FAIL %s\n", strerror(errno)); return 1; }
    double d = now() + 2; while (w[0] != 2 && now() < d) ; if (w[0] != 2) break; spin_ok++; }
  printf("shared_memory_spin: %d/%d, round_trip_us=%.1f\n", spin_ok, N, (now() - t) / (spin_ok ? spin_ok : 1) * 1e6);
  if (spin_ok != N) return 1;
  int wake_ok = 0, timeouts = 0; t = now();
  for (int i = 0; i < N; i++) { w[0] = 1; uint64_t x = 1; write(fds[1], &x, 8);
    while (w[0] != 2) { struct timespec ts = {0, 200000000}; int r = syscall(SYS_futex, w, FUTEX_WAIT, 1, &ts, 0, 0); if (r && errno == ETIMEDOUT) { timeouts++; break; } }
    if (w[0] == 2) wake_ok++; if (timeouts > 5) break; }
  printf("futex_wake_across_boundary: done=%d timeouts_200ms=%d round_trip_us=%.1f\n", wake_ok, timeouts, (now() - t) / (wake_ok ? wake_ok : 1) * 1e6);
  return 0;
}
