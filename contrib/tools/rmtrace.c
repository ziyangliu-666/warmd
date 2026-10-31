// rmtrace.c - LD_PRELOAD shim: decode every NVIDIA driver ioctl + mmap from user space.
// For each call logs: t_start(ns, CLOCK_MONOTONIC), dur(ns), tid, device(ctl/gpu/uvm), escape, class|cmd, status.
// RM_ALLOC -> object class created (channel, TSG, VA space, compute engine, memory...)
// RM_CONTROL -> control command id (decoded offline from SDK headers)
// Build: gcc -O2 -shared -fPIC rmtrace.c -o librmtrace.so -ldl
// Use:   RMTRACE_OUT=/tmp/x.tsv LD_PRELOAD=./librmtrace.so prog
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/mman.h>

#define IOC_NR(r)   ((r) & 0xff)
#define IOC_TYPE(r) (((r) >> 8) & 0xff)
#define IOC_SIZE(r) (((r) >> 16) & 0x3fff)
#define MAXEV 400000
typedef struct { uint64_t t, d; int tid, fdk; unsigned nr, sz, arg, status; uint64_t extra; } ev_t;
static ev_t *E; static volatile int ne; static char fdkind[4096];
static int (*real_ioctl)(int, unsigned long, ...);
static void *(*real_mmap)(void*, size_t, int, int, int, off_t);
static uint64_t ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000000000ull+t.tv_nsec;}

static int kind(int fd){ // 1=nvidiactl 2=nvidiaN 3=uvm 4=other nvidia 9=not nvidia
  if(fd<0||fd>=4096) return 9;
  if(fdkind[fd]) return fdkind[fd];
  char p[64],b[256]; snprintf(p,64,"/proc/self/fd/%d",fd); ssize_t n=readlink(p,b,255); if(n<0) return 9; b[n]=0;
  int k = strstr(b,"nvidiactl")?1: strstr(b,"nvidia-uvm")?3: !strncmp(b,"/dev/nvidia",11)?2:9;
  return fdkind[fd]=k;
}
static void rec(uint64_t t,uint64_t d,int k,unsigned nr,unsigned sz,unsigned arg,unsigned st,uint64_t ex){
  if(!E) return; int i=__sync_fetch_and_add(&ne,1); if(i>=MAXEV) return;
  E[i]=(ev_t){t,d,(int)syscall(SYS_gettid),k,nr,sz,arg,st,ex};
}
int ioctl(int fd, unsigned long req, ...){
  va_list ap; va_start(ap,req); void*a=va_arg(ap,void*); va_end(ap);
  if(!real_ioctl) real_ioctl=dlsym(RTLD_NEXT,"ioctl");
  int k=kind(fd); if(k==9) return real_ioctl(fd,req,a);
  // fdkind cache can go stale when fds are reused; re-check on close is skipped, acceptable for tracing
  unsigned nr=IOC_NR(req), sz=IOC_SIZE(req); void*p=a;
  if(k!=3 && nr==211){ uint32_t*x=a; nr=x[0]; sz=x[1]; p=*(void**)(x+2); } // NV_ESC_IOCTL_XFER_CMD
  if(k==3){ nr=(unsigned)req; sz=0; }
  uint64_t t=ns(); int r=real_ioctl(fd,req,a); uint64_t d=ns()-t;
  unsigned arg=0,st=0; uint64_t ex=0; uint32_t*u=p;
  if(k!=3 && p){
    if(nr==0x2B){ arg=u[3]; st= sz>=48? u[10] : u[7]; ex=u[2];
      if((arg==0x3e||arg==0x40||arg==0x50a0) && *(void**)(u+4)) ex=*(uint64_t*)((char*)*(void**)(u+4)+64); } // memory: size        // RM_ALLOC: class, status, handle
    else if(nr==0x2A){ arg=u[2]; st=u[7]; ex=u[6]; }                    // RM_CONTROL: cmd, status, paramsSize
    else if(nr==0x29){ arg=0; st=u[3]; }                                // RM_FREE
    else if(nr==0x4E){ ex=((uint64_t*)p)[2+1]; st=u[10]; }             // MAP_MEMORY: length
    else if(nr==0x57){ ex=((uint64_t*)p)[3]; }                          // MAP_MEMORY_DMA: length
    else if(nr==0x27){ arg=u[3]; }                                      // ALLOC_MEMORY: class
  } else if(k==3 && p){ st=*(uint32_t*)((char*)p); }
  rec(t,d,k,nr,sz,arg,st,ex); return r;
}
void *mmap(void*addr,size_t len,int prot,int fl,int fd,off_t off){
  if(!real_mmap) return (void*)syscall(SYS_mmap,addr,len,prot,fl,fd,off);
  int k=kind(fd); uint64_t t=ns(); void*r=real_mmap(addr,len,prot,fl,fd,off);
  if(k!=9) rec(t,ns()-t,k,0x10000,0,(unsigned)(off>>12),0,len);
  else if(fd<0 && len>=(1ull<<30)) rec(t,ns()-t,0,0x10001,0,0,0,len); // large anon VA reservations
  return r;
}
static void dump(void){
  const char*o=getenv("RMTRACE_OUT"); FILE*f=o?fopen(o,"w"):stderr; if(!f) return;
  int n=ne<MAXEV?ne:MAXEV;
  for(int i=0;i<n;i++) fprintf(f,"%lu\t%lu\t%d\t%d\t%#x\t%u\t%#x\t%#x\t%lu\n",E[i].t,E[i].d,E[i].tid,E[i].fdk,E[i].nr,E[i].sz,E[i].arg,E[i].status,E[i].extra);
  if(f!=stderr) fclose(f);
}
__attribute__((constructor)) static void init(void){
  real_ioctl=dlsym(RTLD_NEXT,"ioctl"); real_mmap=dlsym(RTLD_NEXT,"mmap");
  E=calloc(MAXEV,sizeof(ev_t)); atexit(dump);
}
