// driver_features.c - driver-API feature checks; run natively and through warmd, compare output.
// Each check prints {"test":..., "ok":...}.
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CK(x) do { CUresult r_ = (x); if (r_) { const char* s; cuGetErrorString(r_, &s); printf("{\"test\":\"%s\",\"ok\":false,\"error\":\"%s -> %s\"}\n", T, #x, s); return 0; } } while (0)
static const char* T;
static int pass(int ok, const char* extra) { printf("{\"test\":\"%s\",\"ok\":%s%s%s}\n", T, ok ? "true" : "false", extra ? "," : "", extra ? extra : ""); return ok; }

// fill(p, v, n): p[i] = v + i ; slow(p, n): spins, then p[i] *= 2  (slow writer to expose ordering bugs)
static const char PTX[] =
".version 7.0\n.target sm_80\n.address_size 64\n"
".visible .entry fill(.param .u64 p, .param .f32 v, .param .u32 n)\n{\n"
" .reg .u64 %a,%o; .reg .u32 %i,%n; .reg .f32 %f,%g; .reg .pred %q;\n"
" ld.param.u64 %a,[p]; ld.param.f32 %f,[v]; ld.param.u32 %n,[n]; mov.u32 %i,%tid.x;\n"
"L: setp.ge.u32 %q,%i,%n; @%q bra D; cvt.rn.f32.u32 %g,%i; add.f32 %g,%g,%f; mul.wide.u32 %o,%i,4; add.u64 %o,%a,%o; st.global.f32 [%o],%g; add.u32 %i,%i,256; bra L;\n"
"D: ret;\n}\n"
".visible .entry slow(.param .u64 p, .param .u32 n)\n{\n"
" .reg .u64 %a,%o,%t0,%t1; .reg .u32 %i,%n; .reg .f32 %g; .reg .pred %q;\n"
" ld.param.u64 %a,[p]; ld.param.u32 %n,[n]; mov.u64 %t0,%globaltimer;\n"
"W: mov.u64 %t1,%globaltimer; sub.u64 %t1,%t1,%t0; setp.lt.u64 %q,%t1,20000000; @%q bra W;\n"
" mov.u32 %i,%tid.x;\n"
"L: setp.ge.u32 %q,%i,%n; @%q bra D; mul.wide.u32 %o,%i,4; add.u64 %o,%a,%o; ld.global.f32 %g,[%o]; add.f32 %g,%g,%g; st.global.f32 [%o],%g; add.u32 %i,%i,256; bra L;\n"
"D: ret;\n}\n";
#define N 4096

int main(void) {
  CUdevice dev; CUcontext ctx; CUmodule m; CUfunction ffill, fslow;
  T = "init"; CK(cuInit(0)); CK(cuDeviceGet(&dev, 0)); CK(cuDevicePrimaryCtxRetain(&ctx, dev)); CK(cuCtxSetCurrent(ctx));
  CK(cuModuleLoadData(&m, PTX)); CK(cuModuleGetFunction(&ffill, m, "fill")); CK(cuModuleGetFunction(&fslow, m, "slow"));
  CUdeviceptr d; CK(cuMemAlloc(&d, N * 4)); unsigned n = N; float h[N];

  { T = "blocking_stream_orders_after_default";   // legacy semantics: blocking stream waits for stream 0
    CUstream b; CK(cuStreamCreate(&b, CU_STREAM_DEFAULT));
    float v = 1; void* a1[] = {&d, &v, &n}; CK(cuLaunchKernel(ffill, 1, 1, 1, 256, 1, 1, 0, 0, a1, 0));
    void* a2[] = {&d, &n}; CK(cuLaunchKernel(fslow, 1, 1, 1, 256, 1, 1, 0, 0, a2, 0));     // default stream: slow doubling
    float w = 100; void* a3[] = {&d, &w, &n}; CK(cuLaunchKernel(ffill, 1, 1, 1, 256, 1, 1, 0, b, a3, 0));  // must run after
    CK(cuStreamSynchronize(b)); CK(cuCtxSynchronize()); CK(cuMemcpyDtoH(h, d, N * 4));
    pass(h[0] == 100 && h[N - 1] == 100 + N - 1, 0); CK(cuStreamDestroy(b)); }

  { T = "nonblocking_stream_does_not_wait";
    CUstream nb; CK(cuStreamCreate(&nb, CU_STREAM_NON_BLOCKING)); CUevent e0, e1; CK(cuEventCreate(&e0, 0)); CK(cuEventCreate(&e1, 0));
    void* a2[] = {&d, &n}; CK(cuLaunchKernel(fslow, 1, 1, 1, 256, 1, 1, 0, 0, a2, 0));
    CK(cuEventRecord(e0, nb)); float w = 5; void* a3[] = {&d, &w, &n}; CUdeviceptr d2; CK(cuMemAlloc(&d2, N * 4)); void* a4[] = {&d2, &w, &n};
    CK(cuLaunchKernel(ffill, 1, 1, 1, 256, 1, 1, 0, nb, a4, 0)); CK(cuEventRecord(e1, nb)); CK(cuEventSynchronize(e1));
    CUresult q = cuStreamQuery(0);   // the slow kernel on stream 0 should still be running
    CK(cuCtxSynchronize()); (void)a3; char x[64]; snprintf(x, 64, "\"default_still_busy\":%s", q == CUDA_ERROR_NOT_READY ? "true" : "false");
    pass(q == CUDA_ERROR_NOT_READY, x); CK(cuMemFree(d2)); }

  { T = "pinned_async_d2h_deferred";
    float* pin; CK(cuMemHostAlloc((void**)&pin, N * 4, 0)); memset(pin, 0, N * 4); CUstream s; CK(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));
    float v = 7; void* a1[] = {&d, &v, &n}; CK(cuLaunchKernel(ffill, 1, 1, 1, 256, 1, 1, 0, s, a1, 0));
    CK(cuMemcpyDtoHAsync(pin, d, N * 4, s)); CUevent e; CK(cuEventCreate(&e, 0)); CK(cuEventRecord(e, s));
    while (cuEventQuery(e) == CUDA_ERROR_NOT_READY) {}
    pass(pin[0] == 7 && pin[N - 1] == 7 + N - 1, 0); CK(cuMemFreeHost(pin)); }

  { T = "memset_d16_d32";
    CK(cuMemsetD32(d, 0x40400000u, N)); CK(cuMemcpyDtoH(h, d, 16)); int ok = h[0] == 3.0f && h[3] == 3.0f;
    CK(cuMemsetD16(d, 0x3c00, 2 * N)); unsigned short hs[4]; CK(cuMemcpyDtoH(hs, d, 8)); pass(ok && hs[0] == 0x3c00 && hs[3] == 0x3c00, 0); }

  { T = "memcpy2d_pitched";
    CUdeviceptr p; size_t pitch; CK(cuMemAllocPitch(&p, &pitch, 100 * 4, 50, 4));
    float src[50][100], dst[50][100]; for (int y = 0; y < 50; y++) for (int x = 0; x < 100; x++) src[y][x] = y * 1000 + x;
    CUDA_MEMCPY2D c = {0}; c.srcMemoryType = CU_MEMORYTYPE_HOST; c.srcHost = src; c.srcPitch = 400; c.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    c.dstDevice = p; c.dstPitch = pitch; c.WidthInBytes = 400; c.Height = 50; CK(cuMemcpy2D(&c));
    CUDA_MEMCPY2D b = {0}; b.srcMemoryType = CU_MEMORYTYPE_DEVICE; b.srcDevice = p; b.srcPitch = pitch; b.srcY = 10; b.dstMemoryType = CU_MEMORYTYPE_HOST;
    b.dstHost = dst; b.dstPitch = 400; b.WidthInBytes = 400; b.Height = 40; CK(cuMemcpy2D(&b));
    char x[64]; snprintf(x, 64, "\"pitch\":%zu", pitch); pass(dst[0][0] == 10000 && dst[39][99] == 49099 && pitch >= 400, x); CK(cuMemFree(p)); }

  { T = "launch_kernel_ex_with_attrs";
    CUlaunchAttribute at[1]; memset(at, 0, sizeof at); at[0].id = CU_LAUNCH_ATTRIBUTE_PRIORITY; at[0].value.priority = 0;
    CUlaunchConfig cfg = {1, 1, 1, 256, 1, 1, 0, 0, at, 1}; float v = 11; void* a1[] = {&d, &v, &n};
    CK(cuLaunchKernelEx(&cfg, ffill, a1, 0)); CK(cuCtxSynchronize()); CK(cuMemcpyDtoH(h, d, 8)); pass(h[1] == 12, 0); }

  { T = "jit_link_ptx_to_cubin";
    CUlinkState ls; CK(cuLinkCreate(0, 0, 0, &ls)); CK(cuLinkAddData(ls, CU_JIT_INPUT_PTX, (void*)PTX, sizeof PTX, "k.ptx", 0, 0, 0));
    void* cub; size_t sz; CK(cuLinkComplete(ls, &cub, &sz)); CUmodule m2; CK(cuModuleLoadData(&m2, cub)); CK(cuLinkDestroy(ls));
    CUfunction f2; CK(cuModuleGetFunction(&f2, m2, "fill")); float v = 3; void* a1[] = {&d, &v, &n};
    CK(cuLaunchKernel(f2, 1, 1, 1, 256, 1, 1, 0, 0, a1, 0)); CK(cuCtxSynchronize()); CK(cuMemcpyDtoH(h, d, 8));
    char x[64]; snprintf(x, 64, "\"cubin_bytes\":%zu", sz); pass(h[1] == 4, x); }

  { T = "graph_capture_driver_api";
    CUstream s; CK(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING)); CUgraph g; CUgraphExec ge;
    CK(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_GLOBAL)); CUstreamCaptureStatus cs; CK(cuStreamIsCapturing(s, &cs));
    float v = 20; void* a1[] = {&d, &v, &n}; CK(cuLaunchKernel(ffill, 1, 1, 1, 256, 1, 1, 0, s, a1, 0));
    CK(cuStreamEndCapture(s, &g)); size_t nn = 0; CK(cuGraphGetNodes(g, 0, &nn));
    CK(cuGraphInstantiate(&ge, g, 0)); CK(cuMemsetD8(d, 0, N * 4)); CK(cuGraphLaunch(ge, s)); CK(cuStreamSynchronize(s));
    CK(cuMemcpyDtoH(h, d, 8)); char x[64]; snprintf(x, 64, "\"nodes\":%zu,\"was_capturing\":%d", nn, cs == CU_STREAM_CAPTURE_STATUS_ACTIVE);
    pass(h[1] == 21 && nn == 1 && cs == CU_STREAM_CAPTURE_STATUS_ACTIVE, x); CK(cuGraphExecDestroy(ge)); CK(cuGraphDestroy(g)); }

  { T = "stream_ordered_alloc";
    CUstream s; CK(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING)); CUdeviceptr q; CK(cuMemAllocAsync(&q, 1 << 20, s));
    CK(cuMemsetD8Async(q, 9, 1 << 20, s)); unsigned char b[4]; CK(cuMemcpyDtoHAsync(b, q, 4, s)); CK(cuStreamSynchronize(s));
    CK(cuMemFreeAsync(q, s)); CK(cuStreamSynchronize(s)); pass(b[0] == 9 && b[3] == 9, 0); }

  { T = "pointer_attributes";
    unsigned mt = 0; CK(cuPointerGetAttribute(&mt, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, d)); CUdeviceptr base; size_t sz;
    CK(cuMemGetAddressRange(&base, &sz, d + 100)); pass(mt == CU_MEMORYTYPE_DEVICE && base == d && sz >= N * 4, 0); }

  { T = "free_null_and_double_free";
    CK(cuMemFree(0)); CUdeviceptr q; CK(cuMemAlloc(&q, 64)); CK(cuMemFree(q)); CUresult r = cuMemFree(q);
    char x[64]; snprintf(x, 64, "\"double_free_err\":%d", r); pass(r != CUDA_SUCCESS, x); }
  return 0;
}
