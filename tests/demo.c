// demo.c - an *unmodified* CUDA driver-API program (linked against the transparent libcuda shim).
// Full lifecycle: init, query, alloc, memcpy, module load (PTX JIT), launch, sync, read back.
// DEMO_BAD=1 loads a faulting kernel instead (to test fault containment through the shim).
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double us(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}
#define CK(x) do{CUresult r_=(x);if(r_){const char*s;cuGetErrorString(r_,&s);fprintf(stderr,"%s -> %s\n",#x,s);exit(1);}}while(0)
static const char PTX[] =
".version 7.0\n.target sm_80\n.address_size 64\n"
".visible .entry vadd(.param .u64 a, .param .u64 b, .param .u64 c, .param .u32 n)\n{\n"
" .reg .u64 %ra,%rb,%rc,%t1,%t2; .reg .u32 %i,%nn; .reg .f32 %fa,%fb; .reg .pred %p;\n"
" ld.param.u64 %ra,[a]; ld.param.u64 %rb,[b]; ld.param.u64 %rc,[c]; ld.param.u32 %nn,[n];\n"
" mov.u32 %i,%tid.x;\n"
"loop: setp.ge.u32 %p,%i,%nn; @%p bra done;\n"
" mul.wide.u32 %t1,%i,4;\n"
" add.u64 %t2,%ra,%t1; ld.global.f32 %fa,[%t2];\n"
" add.u64 %t2,%rb,%t1; ld.global.f32 %fb,[%t2];\n"
" add.f32 %fa,%fa,%fb;\n"
" add.u64 %t2,%rc,%t1; st.global.f32 [%t2],%fa;\n"
" add.u32 %i,%i,1024; bra loop;\n"
"done: ret;\n}\n";
static const char BADPTX[] =
".version 7.0\n.target sm_80\n.address_size 64\n"
".visible .entry bad(.param .u64 a)\n{\n .reg .u64 %r; .reg .u32 %t; mov.u64 %r,8; mov.u32 %t,1; st.global.u32 [%r],%t; ret;\n}\n";
#define N 1024
int main(){
  double t0=us(); CUdevice dev; CUcontext ctx; CUmodule m; CUfunction f; CUdeviceptr da,db,dc; CUstream st; int sms;
  CK(cuInit(0));                                        double t1=us();
  CK(cuDeviceGet(&dev,0));
  CK(cuDeviceGetAttribute(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev));
  CK(cuDevicePrimaryCtxRetain(&ctx,dev)); CK(cuStreamCreate(&st,0)); double t2=us();
  CK(cuMemAlloc(&da,N*4)); CK(cuMemAlloc(&db,N*4)); CK(cuMemAlloc(&dc,N*4));
  float a[N],b[N],c[N]; for(int i=0;i<N;i++){a[i]=i*0.5f;b[i]=i*0.25f;}
  CK(cuMemcpyHtoD(da,a,N*4)); CK(cuMemcpyHtoD(db,b,N*4));
  CK(cuMemsetD8(dc,0,N*4));                             double t3=us();
  int bad = getenv("DEMO_BAD")!=NULL;
  CK(cuModuleLoadData(&m, bad?BADPTX:PTX)); CK(cuModuleGetFunction(&f,m,bad?"bad":"vadd")); double t4=us();
  unsigned n=N; void*args[]={&da,&db,&dc,&n};
  CUresult lr = bad ? cuLaunchKernel(f,1,1,1,1,1,1,0,st,args,0) : cuLaunchKernel(f,1,1,1,1024,1,1,0,st,args,0);
  CUresult sr = cuStreamSynchronize(st);                double t5=us();
  if(!bad){ CK(cuMemcpyDtoH(c,dc,N*4)); long wrong=0; for(int i=0;i<N;i++) if(c[i]!=a[i]+b[i]) wrong++;
    double l0=us(); for(int i=0;i<200;i++){ cuLaunchKernel(f,1,1,1,1024,1,1,0,st,args,0); cuStreamSynchronize(st); } double l1=us();
    CK(cuMemFree(da)); CK(cuMemFree(db)); CK(cuMemFree(dc));
    printf("{\"demo\":1,\"bad\":0,\"wrong\":%ld,\"sms\":%d,\"init_ms\":%.2f,\"ctx_ms\":%.2f,\"alloc_h2d_ms\":%.2f,\"module_ms\":%.2f,"
      "\"launch_sync_ms\":%.2f,\"steady_launch_sync_us\":%.1f,\"total_ms\":%.2f}\n",
      wrong,sms,(t1-t0)/1e3,(t2-t1)/1e3,(t3-t2)/1e3,(t4-t3)/1e3,(t5-t4)/1e3,(l1-l0)/200,(t5-t0)/1e3);
  } else printf("{\"demo\":1,\"bad\":1,\"launch\":%d,\"sync\":%d,\"total_ms\":%.2f}\n",lr,sr,(t5-t0)/1e3);
  return 0;
}
