// common.h - small helpers shared by warmd (supervisor) and its CUDA workers.
#ifndef WD_COMMON_H
#define WD_COMMON_H
#define _GNU_SOURCE
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <linux/futex.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include "../proto/warmd_proto.h"

extern int wd_log_level;          // 0=error 1=warn 2=info 3=debug
extern const char* wd_log_comp;
static inline void wd_log(int lvl, const char* fmt, ...) {
  if (lvl > wd_log_level) return;
  static const char* L[] = {"ERROR", "WARN", "INFO", "DEBUG"};
  struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); struct tm tm; gmtime_r(&ts.tv_sec, &tm);
  char msg[1024]; va_list ap; va_start(ap, fmt); vsnprintf(msg, sizeof msg, fmt, ap); va_end(ap);
  fprintf(stderr, "%04d-%02d-%02dT%02d:%02d:%02d.%03ldZ %-5s [%s %d] %s\n", tm.tm_year + 1900, tm.tm_mon + 1,
          tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, ts.tv_nsec / 1000000, L[lvl], wd_log_comp, getpid(), msg);
}
#define LOGE(...) wd_log(0, __VA_ARGS__)
#define LOGW(...) wd_log(1, __VA_ARGS__)
#define LOGI(...) wd_log(2, __VA_ARGS__)
#define LOGD(...) wd_log(3, __VA_ARGS__)

static inline double wd_now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static inline void wd_futex_wake(volatile int32_t* a) { syscall(SYS_futex, (int*)a, FUTEX_WAKE, 1 << 30, 0, 0, 0); }

// send/recv one message plus up to 4 fds (SCM_RIGHTS)
static inline int wd_send_fds(int sock, const void* buf, size_t n, const int* fds, int nfd) {
  struct iovec io = {(void*)buf, n};
  char cbuf[CMSG_SPACE(sizeof(int) * 4)]; memset(cbuf, 0, sizeof cbuf);
  struct msghdr m = {0}; m.msg_iov = &io; m.msg_iovlen = 1;
  if (nfd > 0) {
    m.msg_control = cbuf; m.msg_controllen = CMSG_SPACE(sizeof(int) * nfd);
    struct cmsghdr* c = CMSG_FIRSTHDR(&m); c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS;
    c->cmsg_len = CMSG_LEN(sizeof(int) * nfd); memcpy(CMSG_DATA(c), fds, sizeof(int) * nfd);
  }
  ssize_t r; do r = sendmsg(sock, &m, MSG_NOSIGNAL); while (r < 0 && errno == EINTR);
  return r == (ssize_t)n ? 0 : -1;
}
static inline ssize_t wd_recv_fds(int sock, void* buf, size_t n, int* fds, int* nfd) {
  struct iovec io = {buf, n};
  char cbuf[CMSG_SPACE(sizeof(int) * 4)];
  struct msghdr m = {0}; m.msg_iov = &io; m.msg_iovlen = 1; m.msg_control = cbuf; m.msg_controllen = sizeof cbuf;
  ssize_t r; do r = recvmsg(sock, &m, MSG_CMSG_CLOEXEC); while (r < 0 && errno == EINTR);
  if (nfd) *nfd = 0;
  if (r > 0 && fds && nfd)
    for (struct cmsghdr* c = CMSG_FIRSTHDR(&m); c; c = CMSG_NXTHDR(&m, c))
      if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) {
        int k = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int); if (k > 4) k = 4;
        memcpy(fds, CMSG_DATA(c), sizeof(int) * k); *nfd = k;
      }
  return r;
}
#endif
