// warmd.c - supervisor for one GPU. It never touches CUDA. It owns:
//   * the unix socket clients connect to (hello / version check / SO_PEERCRED allowlist)
//   * a pool of session regions (memfd + eventfd each), handed to exactly one client at a time
//   * CUDA worker processes: one ACTIVE (serves sessions), K warm STANDBYs (context already built)
//   * session lifetime: client socket hang-up => worker frees the session's GPU resources, region is
//     zeroed (resident pages only) and returned to the pool
//   * failover: if the active worker dies, every open session gets a sticky WD_ERR_DEVICE_LOST
//     (the faulting one gets the fault code) and a standby is promoted in microseconds
//   * /metrics (Prometheus text format) on a local TCP port
#include "common.h"
#include <fcntl.h>
#include <stddef.h>
#include <signal.h>
#include <poll.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/signalfd.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>

int wd_log_level = 2;
const char* wd_log_comp = "warmd";
int worker_main(int argc, char** argv);   // worker.c

// ---------------- configuration ----------------
static struct {
  char socket[256]; int gpu, sessions, rpc_mb, ring_mb, standbys, threads, metrics_port;
  long quota_mb; int allow_all; int nuid; uid_t uids[32]; int socket_mode; int wait_ms;
} cfg = {"/tmp/warmd/gpu0.sock", 0, 64, WD_DEF_RPC_MB, WD_DEF_RING_MB, 1, 4, 0, 0, 0, 0, {0}, 0666, 30000};

static void usage(void) {
  fprintf(stderr,
    "warmd - resident CUDA host for one GPU\n"
    "  --socket PATH       unix socket (default /tmp/warmd/gpu0.sock)\n"
    "  --gpu N             CUDA device ordinal (default 0)\n"
    "  --sessions N        max concurrent client sessions (default 64)\n"
    "  --rpc-mb N          per-session RPC staging (default 8)\n"
    "  --ring-mb N         per-session async copy ring (default 8)\n"
    "  --standbys K        warm standby workers for failover (default 1)\n"
    "  --threads N         dispatch threads in the worker (default 4)\n"
    "  --quota-mb N        device memory limit per session, 0 = none (default 0)\n"
    "  --allow-uid U       allow this uid (repeatable; default: own uid and root)\n"
    "  --allow-all         allow any local uid\n"
    "  --socket-mode OCT   socket file mode (default 0666; access is also checked by uid)\n"
    "  --session-wait-ms N when all sessions are busy, queue new clients up to N ms (default 30000)\n"
    "  --metrics-port P    serve Prometheus metrics on 127.0.0.1:P (default off)\n"
    "  --log-level N       0 error .. 3 debug (default 2)\n"
    "  --config FILE       key = value lines with the same names (without --)\n");
}
static int set_opt(const char* k, const char* v) {
  #define IS(x) (!strcmp(k, x))
  if (IS("socket")) snprintf(cfg.socket, sizeof cfg.socket, "%s", v);
  else if (IS("gpu")) cfg.gpu = atoi(v);
  else if (IS("sessions")) cfg.sessions = atoi(v);
  else if (IS("rpc-mb")) cfg.rpc_mb = atoi(v);
  else if (IS("ring-mb")) cfg.ring_mb = atoi(v);
  else if (IS("standbys")) cfg.standbys = atoi(v);
  else if (IS("threads")) cfg.threads = atoi(v);
  else if (IS("quota-mb")) cfg.quota_mb = atol(v);
  else if (IS("allow-uid")) { if (cfg.nuid < 32) cfg.uids[cfg.nuid++] = (uid_t)atoi(v); }
  else if (IS("allow-all")) cfg.allow_all = !v || !*v || atoi(v) || !strcmp(v, "true");
  else if (IS("socket-mode")) cfg.socket_mode = (int)strtol(v, 0, 8);
  else if (IS("metrics-port")) cfg.metrics_port = atoi(v);
  else if (IS("session-wait-ms")) cfg.wait_ms = atoi(v);
  else if (IS("log-level")) wd_log_level = atoi(v);
  else return -1;
  return 0;
}
static void load_config(const char* path) {
  FILE* f = fopen(path, "r"); if (!f) { LOGE("cannot open config %s: %s", path, strerror(errno)); exit(2); }
  char line[512];
  while (fgets(line, sizeof line, f)) {
    char* p = line; while (*p == ' ' || *p == '\t') p++;
    if (*p == '#' || *p == '\n' || !*p) continue;
    char* eq = strchr(p, '='); char* k = p; char* v = "";
    if (eq) { *eq = 0; v = eq + 1; }
    for (char* e = k + strlen(k); e > k && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\n'); ) *--e = 0;
    while (*v == ' ' || *v == '\t') v++;
    for (char* e = v + strlen(v); e > v && (e[-1] == ' ' || e[-1] == '\n' || e[-1] == '\r'); ) *--e = 0;
    if (set_opt(k, v)) LOGW("config: unknown key '%s'", k);
  }
  fclose(f);
}

// ---------------- state ----------------
enum { R_FREE, R_BUSY };
typedef struct { int memfd, efd, state, cfd; uint32_t sid; pid_t pid; uid_t uid; wd_page_t* page; double t_open; } region_t;
enum { W_SPAWNING, W_STANDBY, W_ACTIVE, W_DEAD };
typedef struct { pid_t pid; int ctl, state, statsfd; wd_stats_t* stats; uint32_t epoch; int fault_region, fault_err; } worker_t;

static region_t* R; static size_t region_size, rpc_off, rpc_size, ring_off, ring_size;
static worker_t W[16]; static int nw; static int active = -1; static uint32_t epoch_ctr, sid_ctr;
static int ep, lsock, msock = -1;
static uint64_t m_sessions_total, m_rejected[8], m_worker_restarts, m_failovers, m_faults;
static double recent_deaths[8]; static int ndeaths;
static wd_stats_t dead_stats;  // accumulated counters of workers that died (keeps /metrics monotonic)

enum { TAG_LISTEN = 1, TAG_SIGNAL, TAG_METRICS, TAG_CLIENT = 1 << 20, TAG_WORKER = 1 << 21 };
static void ep_add(int fd, uint32_t ev, uint64_t tag) { struct epoll_event e = {ev, {.u64 = tag}}; epoll_ctl(ep, EPOLL_CTL_ADD, fd, &e); }

static void ctl_send(int w, uint32_t type, uint32_t region, uint64_t a0, uint64_t a1, const int* fds, int nfd) {
  wd_ctl_t m = {type, region, {a0, a1, 0, 0}};
  if (W[w].state == W_DEAD || wd_send_fds(W[w].ctl, &m, sizeof m, fds, nfd)) LOGW("ctl send to worker %d failed", W[w].pid);
}

// ---------------- workers ----------------
static void spawn_worker(void) {
  int w = -1;
  for (int i = 0; i < nw; i++) if (W[i].state == W_DEAD) { w = i; break; }
  if (w < 0) { if (nw >= 16) { LOGE("too many workers"); return; } w = nw++; }
  int sp[2]; if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sp)) { LOGE("socketpair: %s", strerror(errno)); return; }
  if (W[w].stats) munmap(W[w].stats, sizeof(wd_stats_t));
  if (W[w].statsfd > 0) close(W[w].statsfd);
  int sfd = memfd_create("warmd-stats", MFD_CLOEXEC); if (ftruncate(sfd, sizeof(wd_stats_t))) LOGE("ftruncate stats");
  wd_stats_t* st = mmap(0, sizeof(wd_stats_t), PROT_READ | PROT_WRITE, MAP_SHARED, sfd, 0);
  pid_t pid = fork();
  if (pid == 0) {
    sigset_t none; sigemptyset(&none); sigprocmask(SIG_SETMASK, &none, 0);
    int fd = dup(sp[1]);   // dup clears CLOEXEC
    char fdbuf[16], gpu[16], th[16], ll[16];
    snprintf(fdbuf, 16, "%d", fd); snprintf(gpu, 16, "%d", cfg.gpu); snprintf(th, 16, "%d", cfg.threads); snprintf(ll, 16, "%d", wd_log_level);
    char* argv[] = {"warmd-worker", "--worker", "--ctl-fd", fdbuf, "--gpu", gpu, "--threads", th, "--log-level", ll, 0};
    execv("/proc/self/exe", argv); _exit(127);
  }
  close(sp[1]);
  W[w] = (worker_t){pid, sp[0], W_SPAWNING, sfd, st, ++epoch_ctr, -1, 0};
  ep_add(sp[0], EPOLLIN | EPOLLRDHUP, TAG_WORKER | w);
  // hand the worker every region (it maps and pins them once, at startup) and its stats page
  for (int i = 0; i < cfg.sessions; i++) { int fds[2] = {R[i].memfd, R[i].efd}; ctl_send(w, C_ATTACH, i, region_size, rpc_size, fds, 2); }
  ctl_send(w, C_ATTACH, UINT32_MAX, sizeof(wd_stats_t), 0, &sfd, 1);
  LOGI("spawned worker pid %d (epoch %u)", pid, W[w].epoch);
}
static int count_state(int s) { int n = 0; for (int i = 0; i < nw; i++) n += W[i].state == s; return n; }
static void maybe_promote(void) {
  if (active >= 0) return;
  for (int i = 0; i < nw; i++) if (W[i].state == W_STANDBY) {
    active = i; W[i].state = W_ACTIVE; ctl_send(i, C_PROMOTE, 0, 0, 0, 0, 0);
    LOGI("worker %d promoted to ACTIVE (epoch %u)", W[i].pid, W[i].epoch); return;
  }
}
static void keep_standbys(void) {
  int have = count_state(W_STANDBY) + count_state(W_SPAWNING) + (active >= 0 ? 1 : 0);
  double t = wd_now_us();
  int recent = 0; for (int i = 0; i < ndeaths; i++) recent += t - recent_deaths[i] < 10e6;
  if (recent >= 6) { LOGE("workers crash-looping (%d deaths in 10 s); not respawning for now", recent); return; }
  while (have < 1 + cfg.standbys) { spawn_worker(); have++; }
}
static void zero_region(int i);
static void worker_died(int w, int status) {
  if (W[w].state == W_DEAD) return;
  int was_active = (w == active);
  LOGW("worker %d died (%s %d)%s", W[w].pid, WIFSIGNALED(status) ? "signal" : "exit",
       WIFSIGNALED(status) ? WTERMSIG(status) : WEXITSTATUS(status), was_active ? " - was ACTIVE" : "");
  W[w].state = W_DEAD; epoll_ctl(ep, EPOLL_CTL_DEL, W[w].ctl, 0); close(W[w].ctl);
  if (W[w].stats) {   // fold its counters into the running totals so /metrics stays monotonic
    volatile uint64_t* s = (volatile uint64_t*)W[w].stats; uint64_t* d = (uint64_t*)&dead_stats;
    size_t n = offsetof(wd_stats_t, mem_used) / 8;
    for (size_t k = 0; k < n; k++) d[k] += s[k];
    dead_stats.lib_hits += W[w].stats->lib_hits; dead_stats.lib_miss += W[w].stats->lib_miss;
    dead_stats.launches += W[w].stats->launches; dead_stats.bytes_h2d += W[w].stats->bytes_h2d;
    dead_stats.bytes_d2h += W[w].stats->bytes_d2h; dead_stats.faults += W[w].stats->faults;
    dead_stats.sessions_served += W[w].stats->sessions_served;
  }
  if (ndeaths < 8) recent_deaths[ndeaths++] = wd_now_us(); else { memmove(recent_deaths, recent_deaths + 1, 7 * sizeof(double)); recent_deaths[7] = wd_now_us(); }
  m_worker_restarts++;
  if (was_active) {
    active = -1; m_failovers++;
    int n = 0;
    for (int i = 0; i < cfg.sessions; i++) if (R[i].state == R_BUSY && !R[i].page->dead) {
      int code = (i == W[w].fault_region && W[w].fault_err) ? W[w].fault_err : WD_ERR_DEVICE_LOST;
      R[i].page->dead = code; __sync_synchronize(); R[i].page->state = WD_DONE; R[i].page->err = code;
      wd_futex_wake(&R[i].page->state); n++;
    }
    LOGW("context lost: %d open session(s) notified (culprit region %d)", n, W[w].fault_region);
    maybe_promote();
  }
  keep_standbys();
}

// ---------------- regions ----------------
static void zero_region(int i) {
  // zero only pages that are actually resident (touched) - mincore on our shared mapping
  uint8_t* base = (uint8_t*)R[i].page; size_t pg = 4096, n = region_size / pg;
  static unsigned char* vec; static size_t vn; if (vn < n) { free(vec); vec = malloc(n); vn = n; }
  if (mincore(base, region_size, vec) == 0) { for (size_t k = 0; k < n; k++) if (vec[k] & 1) memset(base + k * pg, 0, pg); }
  else memset(base, 0, region_size);
  uint64_t v; while (read(R[i].efd, &v, 8) == 8) {}   // drain doorbell
}
static void close_session(int i, const char* why) {
  if (R[i].state != R_BUSY) return;
  epoll_ctl(ep, EPOLL_CTL_DEL, R[i].cfd, 0); close(R[i].cfd); R[i].cfd = -1;
  LOGD("session %u (pid %d) closed: %s, %.1f ms", R[i].sid, R[i].pid, why, (wd_now_us() - R[i].t_open) / 1e3);
  if (active >= 0 && !R[i].page->dead) ctl_send(active, C_CLOSE, i, R[i].sid, 0, 0, 0);   // worker frees, then C_CLOSED
  else { zero_region(i); R[i].state = R_FREE; }
}

static int uid_allowed(uid_t u) {
  if (cfg.allow_all || u == 0 || u == getuid()) return 1;
  for (int i = 0; i < cfg.nuid; i++) if (cfg.uids[i] == u) return 1;
  return 0;
}
static void reject(int c, int status, const char* why) {
  wd_welcome_t wl = {WD_MAGIC, WD_PROTO_VERSION, status}; snprintf(wl.reason, sizeof wl.reason, "%s", why);
  wd_send_fds(c, &wl, sizeof wl, 0, 0); close(c); m_rejected[status & 7]++;
  LOGW("rejected client: %s", why);
}
typedef struct { int fd; struct ucred cr; double deadline; } waiter_t;
static waiter_t waitq[4096]; static int nwaitq; static uint64_t m_waited, m_wait_timeouts;
static void serve(int c, struct ucred cr, int i);
static void serve_waiters(void) {
  double now = wd_now_us(); int w = 0;
  for (int k = 0; k < nwaitq; k++) {
    waiter_t x = waitq[k];
    char b; if (recv(x.fd, &b, 1, MSG_DONTWAIT | MSG_PEEK) == 0) { close(x.fd); continue; }   // gave up
    int i = -1;
    if (active >= 0) for (int j = 0; j < cfg.sessions; j++) if (R[j].state == R_FREE) { i = j; break; }
    if (i >= 0) { serve(x.fd, x.cr, i); continue; }
    if (now > x.deadline) { m_wait_timeouts++; reject(x.fd, WD_E_FULL, "all sessions in use (wait timed out)"); continue; }
    waitq[w++] = x;
  }
  nwaitq = w;
}
static void on_accept(void) {
  int c = accept4(lsock, 0, 0, SOCK_CLOEXEC); if (c < 0) return;
  struct timeval tv = {1, 0}; setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  wd_hello_t h; ssize_t n = recv(c, &h, sizeof h, 0);
  if (n != sizeof h || h.magic != WD_MAGIC) return reject(c, WD_E_PROTO, "bad hello");
  if (h.version != WD_PROTO_VERSION) { char b[96]; snprintf(b, 96, "protocol version %u, host speaks %u", h.version, WD_PROTO_VERSION); return reject(c, WD_E_VERSION, b); }
  struct ucred cr; socklen_t cl = sizeof cr; getsockopt(c, SOL_SOCKET, SO_PEERCRED, &cr, &cl);
  if (!uid_allowed(cr.uid)) { char b[96]; snprintf(b, 96, "uid %d not allowed", cr.uid); return reject(c, WD_E_PERM, b); }
  if (active < 0) return reject(c, WD_E_NOWORKER, "no active CUDA worker (starting or crash-looping)");
  int i = -1; for (int k = 0; k < cfg.sessions; k++) if (R[k].state == R_FREE) { i = k; break; }
  if (i < 0) {   // backpressure: park the client until a session frees up (or its wait expires)
    if (nwaitq >= 4096 || cfg.wait_ms <= 0) return reject(c, WD_E_FULL, "all sessions in use");
    waitq[nwaitq++] = (waiter_t){c, cr, wd_now_us() + cfg.wait_ms * 1e3}; m_waited++;
    return;
  }
  serve(c, cr, i);
}
static void serve(int c, struct ucred cr, int i) {
  struct timeval tv;
  R[i].state = R_BUSY; R[i].cfd = c; R[i].sid = ++sid_ctr; R[i].pid = cr.pid; R[i].uid = cr.uid; R[i].t_open = wd_now_us();
  wd_page_t* p = R[i].page; p->magic = WD_MAGIC; p->version = WD_PROTO_VERSION; p->state = WD_IDLE; p->dead = 0;
  p->session_id = R[i].sid; p->epoch = W[active].epoch;
  ctl_send(active, C_OPEN, i, R[i].sid, (uint64_t)cfg.quota_mb << 20, 0, 0);
  wd_welcome_t wl = {WD_MAGIC, WD_PROTO_VERSION, WD_OK, R[i].sid, region_size, rpc_off, rpc_size, ring_off, ring_size,
                     (uint64_t)cfg.quota_mb << 20, 0, cfg.gpu, W[active].epoch, 0, ""};
  int fds[2] = {R[i].memfd, R[i].efd};
  if (wd_send_fds(c, &wl, sizeof wl, fds, 2)) { close_session(i, "welcome failed"); return; }
  tv.tv_sec = 0; tv.tv_usec = 0; setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  ep_add(c, EPOLLIN | EPOLLRDHUP | EPOLLHUP, TAG_CLIENT | i);
  m_sessions_total++;
  LOGD("session %u -> region %d for pid %d uid %d", R[i].sid, i, cr.pid, cr.uid);
}

static void on_worker_msg(int w) {
  wd_ctl_t m; int fds[4], nfd = 0;
  ssize_t n = wd_recv_fds(W[w].ctl, &m, sizeof m, fds, &nfd);
  if (n <= 0) { int st = 0; waitpid(W[w].pid, &st, 0); worker_died(w, st); return; }
  switch (m.type) {
    case C_READY:
      LOGI("worker %d ready in %.0f ms (GPU mem used %.0f MB)", W[w].pid, m.arg[0] / 1e3, m.arg[1] / 1048576.0);
      W[w].state = W_STANDBY; maybe_promote(); break;
    case C_ACTIVE: break;
    case C_CLOSED:
      if (m.region < (uint32_t)cfg.sessions && R[m.region].state == R_BUSY && R[m.region].cfd < 0) { zero_region(m.region); R[m.region].state = R_FREE; }
      break;
    case C_FAULT:
      W[w].fault_region = (int)m.region; W[w].fault_err = (int)m.arg[0]; m_faults++;
      LOGW("worker %d reports sticky fault %d on region %u (session %u)", W[w].pid, (int)m.arg[0], m.region, m.region < (uint32_t)cfg.sessions ? R[m.region].sid : 0);
      break;
  }
}

// ---------------- metrics ----------------
static void on_metrics(void) {
  int c = accept4(msock, 0, 0, SOCK_CLOEXEC); if (c < 0) return;
  char req[1024]; struct timeval tv = {0, 200000}; setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  if (recv(c, req, sizeof req, 0) < 0) {}
  static char body[1 << 16]; size_t o = 0;
  #define M(...) o += snprintf(body + o, sizeof body - o, __VA_ARGS__)
  int busy = 0; for (int i = 0; i < cfg.sessions; i++) busy += R[i].state == R_BUSY;
  wd_stats_t s = dead_stats; if (active >= 0 && W[active].stats) {
    wd_stats_t* a = W[active].stats; s.rpc_total += a->rpc_total; s.rpc_ns += a->rpc_ns;
    for (int k = 0; k < OP_MAX; k++) s.op_count[k] += a->op_count[k];
    s.lib_hits += a->lib_hits; s.lib_miss += a->lib_miss; s.launches += a->launches; s.bytes_h2d += a->bytes_h2d;
    s.bytes_d2h += a->bytes_d2h; s.faults += a->faults; s.sessions_served += a->sessions_served;
    s.mem_used = a->mem_used; s.gpu_mem_total = a->gpu_mem_total; s.gpu_mem_free = a->gpu_mem_free; }
  M("# TYPE warmd_sessions_active gauge\nwarmd_sessions_active %d\n", busy);
  M("warmd_sessions_capacity %d\nwarmd_sessions_waiting %d\nwarmd_sessions_waited_total %lu\nwarmd_sessions_wait_timeouts_total %lu\n",
    cfg.sessions, nwaitq, m_waited, m_wait_timeouts);
  M("# TYPE warmd_sessions_total counter\nwarmd_sessions_total %lu\n", m_sessions_total);
  const char* rs[] = {"", "version", "perm", "full", "noworker", "proto", "", ""};
  for (int k = 1; k < 6; k++) M("warmd_sessions_rejected_total{reason=\"%s\"} %lu\n", rs[k], m_rejected[k]);
  M("warmd_workers{state=\"active\"} %d\nwarmd_workers{state=\"standby\"} %d\nwarmd_workers{state=\"starting\"} %d\n",
    active >= 0, count_state(W_STANDBY), count_state(W_SPAWNING));
  M("warmd_worker_restarts_total %lu\nwarmd_failovers_total %lu\nwarmd_faults_total %lu\n", m_worker_restarts, m_failovers, m_faults);
  M("# TYPE warmd_rpc_total counter\nwarmd_rpc_total %lu\nwarmd_rpc_seconds_total %.6f\n", s.rpc_total, s.rpc_ns / 1e9);
  for (int k = 1; k < OP_MAX; k++) if (s.op_count[k]) M("warmd_rpc_op_total{op=\"%d\"} %lu\n", k, s.op_count[k]);
  M("warmd_kernel_launches_total %lu\nwarmd_copy_bytes_total{dir=\"h2d\"} %lu\nwarmd_copy_bytes_total{dir=\"d2h\"} %lu\n", s.launches, s.bytes_h2d, s.bytes_d2h);
  M("warmd_code_cache_hits_total %lu\nwarmd_code_cache_misses_total %lu\n", s.lib_hits, s.lib_miss);
  M("warmd_session_device_memory_bytes %lu\nwarmd_gpu_memory_total_bytes %lu\nwarmd_gpu_memory_free_bytes %lu\n", s.mem_used, s.gpu_mem_total, s.gpu_mem_free);
  char hdr[160]; int hn = snprintf(hdr, sizeof hdr, "HTTP/1.0 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: %zu\r\n\r\n", o);
  if (write(c, hdr, hn) < 0 || write(c, body, o) < 0) {}
  close(c);
}

static void mkdirs(const char* path) {
  char d[256]; snprintf(d, sizeof d, "%s", path); char* s = strrchr(d, '/'); if (!s || s == d) return; *s = 0;
  for (char* p = d + 1; *p; p++) if (*p == '/') { *p = 0; mkdir(d, 0755); *p = '/'; }
  mkdir(d, 0755);
}

int main(int argc, char** argv) {
  if (argc > 1 && !strcmp(argv[1], "--worker")) return worker_main(argc, argv);
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(); return 0; }
    if (!strcmp(argv[i], "--config") && i + 1 < argc) { load_config(argv[++i]); continue; }
    if (!strncmp(argv[i], "--", 2)) {
      const char* k = argv[i] + 2;
      if (!strcmp(k, "allow-all")) { cfg.allow_all = 1; continue; }
      if (i + 1 >= argc || set_opt(k, argv[i + 1])) { fprintf(stderr, "bad option %s\n", argv[i]); usage(); return 2; }
      i++; continue;
    }
    fprintf(stderr, "unexpected argument %s\n", argv[i]); usage(); return 2;
  }
  if (cfg.sessions < 1 || cfg.sessions > WD_MAX_SESSIONS) { LOGE("--sessions must be 1..%d", WD_MAX_SESSIONS); return 2; }
  rpc_off = 0; rpc_size = (size_t)cfg.rpc_mb << 20; ring_off = rpc_size; ring_size = (size_t)cfg.ring_mb << 20;
  region_size = WD_PAGE + rpc_size + ring_size;

  R = calloc(cfg.sessions, sizeof *R);
  for (int i = 0; i < cfg.sessions; i++) {
    R[i].memfd = memfd_create("warmd-session", MFD_CLOEXEC);
    if (R[i].memfd < 0 || ftruncate(R[i].memfd, region_size)) { LOGE("memfd: %s", strerror(errno)); return 1; }
    R[i].efd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    R[i].page = mmap(0, region_size, PROT_READ | PROT_WRITE, MAP_SHARED, R[i].memfd, 0);
    if (R[i].page == MAP_FAILED) { LOGE("mmap: %s", strerror(errno)); return 1; }
    R[i].cfd = -1;
  }
  ep = epoll_create1(EPOLL_CLOEXEC);
  sigset_t ss; sigemptyset(&ss); sigaddset(&ss, SIGCHLD); sigaddset(&ss, SIGTERM); sigaddset(&ss, SIGINT); sigaddset(&ss, SIGHUP);
  sigprocmask(SIG_BLOCK, &ss, 0); signal(SIGPIPE, SIG_IGN);
  int sfd = signalfd(-1, &ss, SFD_CLOEXEC | SFD_NONBLOCK); ep_add(sfd, EPOLLIN, TAG_SIGNAL);

  mkdirs(cfg.socket); unlink(cfg.socket);
  lsock = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  struct sockaddr_un a = {AF_UNIX}; snprintf(a.sun_path, sizeof a.sun_path, "%s", cfg.socket);
  if (bind(lsock, (struct sockaddr*)&a, sizeof a) || listen(lsock, 256)) { LOGE("bind %s: %s", cfg.socket, strerror(errno)); return 1; }
  chmod(cfg.socket, cfg.socket_mode);
  ep_add(lsock, EPOLLIN, TAG_LISTEN);
  if (cfg.metrics_port) {
    msock = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0); int one = 1; setsockopt(msock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in ma = {AF_INET, htons(cfg.metrics_port), {htonl(INADDR_LOOPBACK)}};
    if (bind(msock, (struct sockaddr*)&ma, sizeof ma) || listen(msock, 16)) { LOGE("metrics bind: %s", strerror(errno)); return 1; }
    ep_add(msock, EPOLLIN, TAG_METRICS);
  }
  LOGI("listening on %s: gpu %d, %d sessions x %.1f MB, %d standby, quota %ld MB, protocol v%u",
       cfg.socket, cfg.gpu, cfg.sessions, region_size / 1048576.0, cfg.standbys, cfg.quota_mb, WD_PROTO_VERSION);
  keep_standbys();

  int stopping = 0;
  while (!stopping) {
    struct epoll_event ev[64]; int n = epoll_wait(ep, ev, 64, nwaitq ? 20 : 1000);
    for (int k = 0; k < n; k++) {
      uint64_t t = ev[k].data.u64;
      if (t == TAG_LISTEN) on_accept();
      else if (t == TAG_METRICS) on_metrics();
      else if (t == TAG_SIGNAL) {
        struct signalfd_siginfo si;
        while (read(sfd, &si, sizeof si) == sizeof si) {
          if (si.ssi_signo == SIGCHLD) {
            int st; pid_t p;
            while ((p = waitpid(-1, &st, WNOHANG)) > 0)
              for (int w = 0; w < nw; w++) if (W[w].pid == p && W[w].state != W_DEAD) worker_died(w, st);
          } else stopping = 1;
        }
      } else if (t & TAG_WORKER) on_worker_msg((int)(t & 0xffff));
      else if (t & TAG_CLIENT) {
        int i = (int)(t & 0xffff); char b[64];
        if (R[i].state == R_BUSY && R[i].cfd >= 0) {
          ssize_t r = recv(R[i].cfd, b, sizeof b, MSG_DONTWAIT);
          if (r == 0 || (r < 0 && errno != EAGAIN) || (ev[k].events & (EPOLLHUP | EPOLLRDHUP))) close_session(i, "client hung up");
        }
      }
    }
    if (nwaitq) serve_waiters();
  }
  LOGI("shutting down");
  for (int w = 0; w < nw; w++) if (W[w].state != W_DEAD) { ctl_send(w, C_STOP, 0, 0, 0, 0, 0); }
  for (int w = 0; w < nw; w++) if (W[w].state != W_DEAD) { int st; for (int t = 0; t < 30 && !waitpid(W[w].pid, &st, WNOHANG); t++) usleep(100000); kill(W[w].pid, SIGKILL); }
  unlink(cfg.socket);
  return 0;
}
