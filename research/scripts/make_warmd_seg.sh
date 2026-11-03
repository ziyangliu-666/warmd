#!/usr/bin/env bash
# Build a copy of warmd with persistent segments. The repo sources stay unchanged.
# A segment is a device allocation owned by the worker, not by a session, so it survives client
# exit. Clients find it by key. Every session in the daemon can read and write it: a warmd daemon is
# one trust domain.
set -e; rm -rf /workspace/warmd-seg; cp -r /workspace/warmd /workspace/warmd-seg; cd /workspace/warmd-seg; rm -rf build
python3 -I - <<'PY'
p = open("src/proto.h").read()
p = p.replace("  OP_MAX\n", "  OP_SEG_OPEN,\n  OP_MAX\n"); open("src/proto.h", "w").write(p)
w = open("src/worker.c").read()
w = w.replace("// ---------------- per-session state ----------------", """static pthread_mutex_t segmu = PTHREAD_MUTEX_INITIALIZER;
static struct { uint64_t k, p, n; } SEG[1024]; static int nseg;
// ---------------- per-session state ----------------""", 1)
w = w.replace("    case OP_MEM_ALLOC: {", """    case OP_SEG_OPEN: {
      pthread_mutex_lock(&segmu); int j = 0; for (; j < nseg; j++) if (SEG[j].k == a[0]) break;
      if (j < nseg) { r[0] = SEG[j].p; r[1] = 0; }
      else { CUdeviceptr q = 0; z = cuMemAlloc(&q, a[1]); if (!z) { SEG[nseg].k = a[0]; SEG[nseg].p = q; SEG[nseg].n = a[1]; nseg++; r[0] = q; r[1] = 1; } }
      pthread_mutex_unlock(&segmu); break; }
    case OP_MEM_ALLOC: {""", 1)
open("src/worker.c", "w").write(w)
s = open("src/shim.c").read()
s = s.replace("CUresult cuMemAlloc_v2(CUdeviceptr* p, size_t n) {", """CUresult cuWarmdSegOpen(uint64_t key, size_t n, uint64_t* p, int* created) { ENTER; A; a[0] = key; a[1] = n; int e = rpc(OP_SEG_OPEN, a, r); *p = r[0]; *created = (int)r[1]; if (!e) { LK; if (!rg_find(r[0])) rg_add(r[0], n, 0); } return e; }
CUresult cuMemAlloc_v2(CUdeviceptr* p, size_t n) {""", 1)
open("src/shim.c", "w").write(s)
PY
make -s 2>&1 | tail -3; ls -la build/warmd build/lib/libcuda.so.1 | cut -c20-
