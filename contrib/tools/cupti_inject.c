// cupti_inject.c - zero-code-change tracer. Load into ANY CUDA process via:
//   CUDA_INJECTION64_PATH=./libcupti_inject.so python task.py
// Reports: wall time, time inside each Driver API call, GPU busy time (kernels+memcpy),
// and "duty" = GPU busy / process wall. Output: one JSON line on stderr at exit.
// Build: gcc -O2 -shared -fPIC cupti_inject.c -o libcupti_inject.so \
//   -I/usr/local/cuda/include -I/usr/local/cuda/extras/CUPTI/include \
//   -L/usr/local/cuda/extras/CUPTI/lib64 -lcupti -lpthread
#include <cupti.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#define MAXCB 1024
static double tot_ms[MAXCB]; static long cnt[MAXCB]; static const char* name[MAXCB];
static __thread double t_enter[64]; static __thread int depth;
static double t_start; static unsigned long long gpu_ns, first_gpu=~0ULL, last_gpu;
static long nkern, ncopy; static FILE*seq; /* CUPTI_SEQ=<file>: ordered call log */ static pthread_mutex_t mu=PTHREAD_MUTEX_INITIALIZER;
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}

static void CUPTIAPI cb(void*u,CUpti_CallbackDomain d,CUpti_CallbackId id,const void*p){
  const CUpti_CallbackData*ci=p; if(id>=MAXCB) return;
  if(ci->callbackSite==CUPTI_API_ENTER){ if(depth<64) t_enter[depth]=now_ms(); depth++;
    if(seq) fprintf(seq,"%s\n",ci->functionName); }
  else { depth--; if(depth<0||depth>=64){depth=depth<0?0:depth;return;}
    double dt=now_ms()-t_enter[depth];
    if(seq && ci->functionReturnValue && *(int*)ci->functionReturnValue) fprintf(seq,"  !! %s -> %d\n",ci->functionName,*(int*)ci->functionReturnValue);
    if(seq && !strcmp(ci->functionName,"cuModuleGetGlobal_v2")){ const struct{unsigned long long*p;size_t*n;void*m;const char*name;}*a=ci->functionParams;
      fprintf(seq,"  -> %s ptr=%#llx size=%zu ret=%d\n",a->name,a->p?*a->p:0,a->n?*a->n:0,*(int*)ci->functionReturnValue); }
    pthread_mutex_lock(&mu); tot_ms[id]+=dt; cnt[id]++; name[id]=ci->functionName; pthread_mutex_unlock(&mu); }
}
static void CUPTIAPI buf_req(uint8_t**b,size_t*s,size_t*m){*s=1<<20;*b=aligned_alloc(8,*s);*m=0;}
static void CUPTIAPI buf_done(CUcontext c,uint32_t sid,uint8_t*b,size_t s,size_t valid){
  CUpti_Activity*r=NULL;
  while(cuptiActivityGetNextRecord(b,valid,&r)==CUPTI_SUCCESS){
    unsigned long long st=0,en=0;
    if(r->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL||r->kind==CUPTI_ACTIVITY_KIND_KERNEL){
      CUpti_ActivityKernel4*k=(void*)r; st=k->start; en=k->end; nkern++; }
    else if(r->kind==CUPTI_ACTIVITY_KIND_MEMCPY){ CUpti_ActivityMemcpy*m=(void*)r; st=m->start; en=m->end; ncopy++; }
    else continue;
    gpu_ns+=en-st; if(st<first_gpu) first_gpu=st; if(en>last_gpu) last_gpu=en;
  }
  free(b);
}
static int cmp(const void*a,const void*b){double x=tot_ms[*(int*)a],y=tot_ms[*(int*)b];return x<y?1:x>y?-1:0;}
static void report(void){
  cuptiActivityFlushAll(1);
  double wall=now_ms()-t_start; int idx[MAXCB]; for(int i=0;i<MAXCB;i++)idx[i]=i;
  qsort(idx,MAXCB,sizeof(int),cmp); double api=0; for(int i=0;i<MAXCB;i++)api+=tot_ms[i];
  fprintf(stderr,"{\"cupti\":1,\"wall_ms\":%.1f,\"driver_api_ms\":%.1f,\"gpu_busy_ms\":%.3f,\"duty\":%.5f,"
    "\"kernels\":%ld,\"copies\":%ld,\"top\":[",wall,api,gpu_ns/1e6,gpu_ns/1e6/wall,nkern,ncopy);
  int top=getenv("CUPTI_ALL")?MAXCB:12;
  for(int i=0;i<top&&tot_ms[idx[i]]>0;i++) fprintf(stderr,"%s[\"%s\",%ld,%.2f]",i?",":"",name[idx[i]],cnt[idx[i]],tot_ms[idx[i]]);
  fprintf(stderr,"]}\n");
}
int InitializeInjection(void){
  t_start=now_ms(); CUpti_SubscriberHandle h;
  if(getenv("CUPTI_SEQ")){ seq=fopen(getenv("CUPTI_SEQ"),"w"); if(seq) setvbuf(seq,0,_IOLBF,0); }
  cuptiSubscribe(&h,(CUpti_CallbackFunc)cb,NULL);
  cuptiEnableDomain(1,h,CUPTI_CB_DOMAIN_DRIVER_API);
  cuptiActivityRegisterCallbacks(buf_req,buf_done);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
  atexit(report); return 1;
}
