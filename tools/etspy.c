// etspy.c - LD_PRELOAD spy for hidden driver export tables (cuGetExportTable).
// Wraps every table listed in ETSPY (comma-separated uuids, or "all") with logging trampolines:
// each call logs table, index, first 6 args and the return value, then forwards to the real driver.
// It is built as a proxy libcuda.so.1 whose DT_NEEDED is the real driver by absolute path, so
// dlopen("libcuda.so.1")+dlsym finds cuGetExportTable/cuGetProcAddress here and everything else
// in the real driver (see bench/etspy.sh).
// Use:   ETSPY=all LD_LIBRARY_PATH=<proxy dir> python3 task.py 2> spy.log
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>

typedef uint64_t u64;
typedef u64 (*fn6)(u64,u64,u64,u64,u64,u64);
#define MAXT 16
#define MAXE 128
static struct { char uuid[33]; const void*real; void*wrap[MAXE]; int n; } T[MAXT]; static int nt;
static fn6 REAL[MAXT][MAXE];
static pthread_mutex_t mu=PTHREAD_MUTEX_INITIALIZER;
static int (*real_get)(const void**,const void*);
static int (*real_gpa)(const char*,void**,int,u64,int*);

static void dump(const char*tag,u64 p);
static void dump(const char*tag,u64 p){ if(p<0x10000||(p>>48)) return; const u64*q=(const u64*)p;
  fprintf(stderr,"[etspy]    %s %#lx:",tag,p); for(int k=0;k<8;k++) fprintf(stderr," %016lx",q[k]); fprintf(stderr,"\n"); }
static u64 spy(int t,int i,u64 a,u64 b,u64 c,u64 d,u64 e,u64 f){
  int dm=getenv("ETSPY_DUMP")!=0;
  if(dm){ dump("in  a0",a); dump("in  a1",b); }
  u64 r=REAL[t][i](a,b,c,d,e,f);
  if(dm){ dump("out a0",a); dump("out a1",b); }
  if(getenv("ETSPY_STR")){ u64 v[6]={a,b,c,d,e,f}; for(int k=0;k<6;k++){ Dl_info di; if(v[k]>0x10000 && !(v[k]>>48) && dladdr((void*)v[k],&di) && di.dli_fname){
        const char*str=(const char*)v[k]; int ok=1; for(int j=0;j<4;j++) if(str[j]<32||str[j]>126) ok=0;
        fprintf(stderr,"[etspy]    a%d in %s+%#lx%s%.80s%s\n",k,strrchr(di.dli_fname,'/')?strrchr(di.dli_fname,'/')+1:di.dli_fname,v[k]-(u64)di.dli_fbase,ok?" \"":"",ok?str:"",ok?"\"":""); } } }
  fprintf(stderr,"[etspy] %s[%d](%#lx,%#lx,%#lx,%#lx,%#lx,%#lx) -> %#lx\n",T[t].uuid,i,a,b,c,d,e,f,r);
  return r; }
#define W(t,n,k) static u64 w_##t##_##n(u64 a,u64 b,u64 c,u64 d,u64 e,u64 f){ return spy(t,k,a,b,c,d,e,f); }
#define W8(t,i) W(t,i##0,(i)*8+0) W(t,i##1,(i)*8+1) W(t,i##2,(i)*8+2) W(t,i##3,(i)*8+3) W(t,i##4,(i)*8+4) W(t,i##5,(i)*8+5) W(t,i##6,(i)*8+6) W(t,i##7,(i)*8+7)
#define WT(t) W8(t,0) W8(t,1) W8(t,2) W8(t,3) W8(t,4) W8(t,5) W8(t,6) W8(t,7) W8(t,8) W8(t,9) W8(t,10) W8(t,11) W8(t,12) W8(t,13) W8(t,14) W8(t,15)
WT(0) WT(1) WT(2) WT(3) WT(4) WT(5) WT(6) WT(7)
#define P(t,i) w_##t##_##i,
#define P8(t,i) P(t,i##0) P(t,i##1) P(t,i##2) P(t,i##3) P(t,i##4) P(t,i##5) P(t,i##6) P(t,i##7)
#define PT(t) { P8(t,0) P8(t,1) P8(t,2) P8(t,3) P8(t,4) P8(t,5) P8(t,6) P8(t,7) P8(t,8) P8(t,9) P8(t,10) P8(t,11) P8(t,12) P8(t,13) P8(t,14) P8(t,15) }
static void*WR[8][MAXE];
static void build(void){
  static void*raw[8][128]={ PT(0), PT(1), PT(2), PT(3), PT(4), PT(5), PT(6), PT(7) };
  memcpy(WR,raw,sizeof raw); }

static int want(const char*h){ const char*e=getenv("ETSPY"); if(!e) return 0; if(!strcmp(e,"all")) return 1;
  char b[512]; snprintf(b,sizeof b,"%s",e); for(char*t=strtok(b,",");t;t=strtok(0,",")) if(!strncmp(h,t,strlen(t))) return 1; return 0; }

int cuGetExportTable(const void**out,const void*id){
  static int built; if(!built){ built=1; build(); }
  if(!real_get) real_get=dlsym(RTLD_NEXT,"cuGetExportTable");
  int e=real_get(out,id); if(e||!*out) return e;
  char h[33]; for(int i=0;i<16;i++) sprintf(h+2*i,"%02x",((const unsigned char*)id)[i]);
  pthread_mutex_lock(&mu);
  for(int t=0;t<nt;t++) if(T[t].real==*out){ *out=T[t].wrap; pthread_mutex_unlock(&mu); return 0; }
  if(!want(h)||nt>=8){ pthread_mutex_unlock(&mu); return 0; }
  int t=nt++; strcpy(T[t].uuid,h); T[t].real=*out;
  const u64*q=*out; int sized = q[0]<0x1000 && q[0]%8==0;   // first slot is the byte size in most tables
  int n = sized ? (int)(q[0]/8) : 4; if(n>MAXE) n=MAXE; T[t].n=n;
  for(int k=0;k<n;k++){ if(k==0&&sized){ T[t].wrap[0]=(void*)q[0]; continue; }
    if(!q[k]){ T[t].wrap[k]=0; continue; }
    REAL[t][k]=(fn6)q[k]; T[t].wrap[k]=WR[t][k]; }
  fprintf(stderr,"[etspy] wrapping table %s (%d entries, sized=%d)\n",h,n,sized);
  *out=T[t].wrap; pthread_mutex_unlock(&mu); return 0; }

int cuGetProcAddress_v2(const char*s,void**p,int v,u64 f,int*st){
  if(!real_gpa) real_gpa=dlsym(RTLD_NEXT,"cuGetProcAddress_v2");
  int e=real_gpa(s,p,v,f,st);
  if(!e && !strcmp(s,"cuGetExportTable")) *p=(void*)cuGetExportTable;
  if(!e && !strcmp(s,"cuGetProcAddress")) *p=(void*)cuGetProcAddress_v2;
  return e; }

int cuGetProcAddress(const char*s,void**p,int v,u64 f){ return cuGetProcAddress_v2(s,p,v,f,0); }
