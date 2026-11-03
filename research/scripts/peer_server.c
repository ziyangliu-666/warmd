// H2: the kernel primitives warmd needs across a sandbox boundary, without a GPU.
// The server runs outside the sandbox, as the warmd supervisor and worker do.
#define _GNU_SOURCE
#include <linux/futex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>
int main(int c, char** v) {
  int s = socket(AF_UNIX, SOCK_STREAM, 0); struct sockaddr_un a = {AF_UNIX}; strcpy(a.sun_path, v[1]); unlink(v[1]);
  if (bind(s, (void*)&a, sizeof a) || listen(s, 1)) { perror("bind"); return 1; } puts("listening"); fflush(stdout);
  int f = accept(s, 0, 0); struct ucred u; socklen_t l = sizeof u; getsockopt(f, SOL_SOCKET, SO_PEERCRED, &u, &l); printf("peer pid=%d uid=%d\n", u.pid, u.uid); fflush(stdout);
  int m = memfd_create("r", 0); ftruncate(m, 1 << 20); int e = eventfd(0, 0); volatile int32_t* w = mmap(0, 1 << 20, PROT_READ | PROT_WRITE, MAP_SHARED, m, 0);
  char b = 'x'; struct iovec io = {&b, 1}; char cb[CMSG_SPACE(8)]; struct msghdr h = {.msg_iov = &io, .msg_iovlen = 1, .msg_control = cb, .msg_controllen = sizeof cb};
  struct cmsghdr* cm = CMSG_FIRSTHDR(&h); cm->cmsg_level = SOL_SOCKET; cm->cmsg_type = SCM_RIGHTS; cm->cmsg_len = CMSG_LEN(8); int fds[2] = {m, e}; memcpy(CMSG_DATA(cm), fds, 8);
  if (sendmsg(f, &h, 0) != 1) { perror("sendmsg"); return 1; } puts("fds sent"); fflush(stdout);
  for (;;) { uint64_t x; if (read(e, &x, 8) != 8) break; if (w[0] != 1) { printf("server: state %d, expected 1\n", w[0]); } w[0] = 2; syscall(SYS_futex, w, FUTEX_WAKE, 1, 0, 0, 0); }
  return 0;
}
