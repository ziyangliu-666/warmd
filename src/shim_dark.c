// shim_dark.c: undocumented driver tables returned by cuGetExportTable. Included by shim.c.
// cudart and cuBLASLt use them. Table layouts and the integrity hash follow ZLUDA
// (github.com/vosen/ZLUDA, MIT or Apache-2.0). The TOOLS_TLS and 21318c60 entries were found with
// contrib/tools/etspy.c on driver 570.211.01.
static int ld_mod(H*m,const void*img){ return cuModuleLoadData(m,img); }
static CUresult d_get_module_from_cubin(H*m,const void*w){ if(!m||!w) return E_INVALID_VALUE; return ld_mod(m,w); }
static CUresult d_cudart_fn2(H*c,CUdevice d){ (void)d; LK; if(ensure_init()) return E_NOT_INITIALIZED; if(!c) return E_INVALID_VALUE; *c=FAKE_CTX; return OK; }
static CUresult d_primary_retain_flags(CUdevice d,unsigned f){ (void)f; H c; return cuDevicePrimaryCtxRetain(&c,d); }
static CUresult d_get_module_ext1(H*m,const void*w,void*a3,void*a4,unsigned a5){
  if(a3||a4||a5) fprintf(stderr,"[warmd] get_module_from_cubin_ext1 extra args %p %p %u ignored\n",a3,a4,a5);
  if(!m||!w) return E_INVALID_VALUE; return ld_mod(m,w); }
static void d_fn7(size_t x){ (void)x; }
static CUresult d_get_module_ext2(const void*fb,H*m,void*a3,void*a4,unsigned a5){
  if(a3||a4||a5) fprintf(stderr,"[warmd] get_module_from_cubin_ext2 extra args %p %p %u ignored\n",a3,a4,a5);
  if(!m||!fb) return E_INVALID_VALUE; return ld_mod(m,fb); }
static CUresult d_load_compilers(void){ return OK; }
static uint32_t UNKBUF1[1024], UNKBUF2[14];
// TOOLS_TLS is 3 slots on driver 570 (size 0x18). [2](void**out,int flag,..) returns a tools object that
// cuBLASLt then passes to 21318c60[4]. We hand out a dummy object (our [4] ignores it).
static uint64_t TOOLS_OBJ[64]={0x100000000ull,0x300000003ull};
static CUresult d_tools_tls1(void){ fprintf(stderr,"[warmd] TOOLS_TLS[1] called -> NOT_SUPPORTED\n"); return E_NOT_SUPPORTED; }
static CUresult d_tools_tls2(void**out,uint64_t flag){ (void)flag; if(out) *out=TOOLS_OBJ; return OK; }
static void d_unk_buf1(void**p,size_t*n){ *p=UNKBUF1; *n=1024; }
static void d_unk_buf2(void**p,size_t*n){ *p=UNKBUF2; *n=14; }
typedef void(*cls_dtor_t)(H,void*,void*);
typedef struct { void*key,*val; cls_dtor_t dtor; } cls_t; static cls_t CLS[256]; static int ncls;
static CUresult d_cls_put(H c,void*k,void*v,cls_dtor_t dt){ (void)c; LK; for(int i=0;i<ncls;i++) if(CLS[i].key==k){ CLS[i].val=v; CLS[i].dtor=dt; return OK; }
  if(ncls>=256) return 2; CLS[ncls++]=(cls_t){k,v,dt}; return OK; }
static CUresult d_cls_del(H c,void*k){ (void)c; LK; for(int i=0;i<ncls;i++) if(CLS[i].key==k){ CLS[i]=CLS[--ncls]; return OK; } return OK; }
static CUresult d_cls_get(void**v,H c,void*k){ (void)c; LK; for(int i=0;i<ncls;i++) if(CLS[i].key==k){ *v=CLS[i].val; return OK; } return 400; /*INVALID_HANDLE*/ }
static CUresult d_ctx_create_bypass(H*c,unsigned f,CUdevice d){ return cuCtxCreate_v2(c,f,d); }
typedef struct { size_t x,y; } heaprec_t;
static CUresult d_heap_alloc(void**rec,size_t a2,size_t a3){ heaprec_t*r=malloc(sizeof *r); r->x=a2; r->y=a3; *rec=r; return OK; }
static CUresult d_heap_free(void*rec,size_t*a2){ heaprec_t*r=rec; if(a2) *a2=r->y; free(r); return OK; }
static CUresult d_dev_attr_ext(CUdevice d,unsigned at,int u,size_t*res){ (void)d;(void)u;(void)res;
  fprintf(stderr,"[warmd] device_get_attribute_ext(%u) -> NOT_SUPPORTED\n",at); return E_NOT_SUPPORTED; }
static CUresult d_dev_something(unsigned char*r,CUdevice d){ (void)d; *r=0; return OK; }
static CUresult d_context_check(H c,uint32_t*r1,const void**r2){ (void)c;(void)r2; *r1=0; return OK; }
static uint32_t d_check_fn3(void){ return 0; }
static CUresult d_integrity_check(uint32_t version,uint64_t secs,uint64_t*res);

// {21318c60-...}: 92-entry table used by cuBLASLt heuristics. Natively cuBLASLt only calls [4](obj,u64*out,0,1,..)
// which sets *out=1 (a capability query). Other entries log their index and fail.
static CUresult d21_unk(int i){ fprintf(stderr,"[warmd] export table 21318c60[%d] called -> NOT_SUPPORTED\n",i); return E_NOT_SUPPORTED; }
#define U21(i) static CUresult d21_##i(void){ return d21_unk(i); }
#define U21x10(p) U21(p##0) U21(p##1) U21(p##2) U21(p##3) U21(p##4) U21(p##5) U21(p##6) U21(p##7) U21(p##8) U21(p##9)
U21(1) U21(2) U21(3) U21(5) U21(6) U21(7) U21(8) U21(9) U21x10(1) U21x10(2) U21x10(3) U21x10(4) U21x10(5) U21x10(6) U21x10(7) U21x10(8) U21(90) U21(91)
static CUresult d21_4(void*obj,uint64_t*out,uint64_t a2,uint64_t a3){ (void)obj;(void)a2;(void)a3; if(out) *out=1; return OK; }
static void*T_21318[92];
static void t21_init(void){
  #define F21(i) T_21318[i]=(void*)d21_##i;
  #define F21x10(p) F21(p##0) F21(p##1) F21(p##2) F21(p##3) F21(p##4) F21(p##5) F21(p##6) F21(p##7) F21(p##8) F21(p##9)
  T_21318[0]=(void*)(uintptr_t)sizeof T_21318; F21(1) F21(2) F21(3) F21(4) F21(5) F21(6) F21(7) F21(8) F21(9)
  F21x10(1) F21x10(2) F21x10(3) F21x10(4) F21x10(5) F21x10(6) F21x10(7) F21x10(8) F21(90) F21(91) }
static void*T_CUDART[13], *T_TOOLS_TLS[3], *T_TOOLS_HOOKS[7], *T_CLS[4], *T_BYPASS[2], *T_HEAP[3], *T_DEVEXT[26], *T_INTEG[3], *T_CHECKS[4];
#define SZ(t) (void*)(uintptr_t)sizeof(t)
static void dark_init(void){ static int done; if(done) return; done=1;
  T_CUDART[0]=SZ(T_CUDART); T_CUDART[1]=(void*)d_get_module_from_cubin; T_CUDART[2]=(void*)d_cudart_fn2; T_CUDART[4]=(void*)d_primary_retain_flags;
  T_CUDART[6]=(void*)d_get_module_ext1; T_CUDART[7]=(void*)d_fn7; T_CUDART[8]=(void*)d_get_module_ext2; T_CUDART[12]=(void*)d_load_compilers;
  T_TOOLS_TLS[0]=SZ(T_TOOLS_TLS); T_TOOLS_TLS[1]=(void*)d_tools_tls1; T_TOOLS_TLS[2]=(void*)d_tools_tls2;
  T_TOOLS_HOOKS[0]=SZ(T_TOOLS_HOOKS); T_TOOLS_HOOKS[2]=(void*)d_unk_buf1; T_TOOLS_HOOKS[6]=(void*)d_unk_buf2;
  T_CLS[0]=(void*)d_cls_put; T_CLS[1]=(void*)d_cls_del; T_CLS[2]=(void*)d_cls_get;
  T_BYPASS[0]=SZ(T_BYPASS); T_BYPASS[1]=(void*)d_ctx_create_bypass;
  T_HEAP[0]=SZ(T_HEAP); T_HEAP[1]=(void*)d_heap_alloc; T_HEAP[2]=(void*)d_heap_free;
  T_DEVEXT[0]=SZ(T_DEVEXT); T_DEVEXT[5]=(void*)d_dev_attr_ext; T_DEVEXT[13]=(void*)d_dev_something;
  T_INTEG[0]=SZ(T_INTEG); T_INTEG[1]=(void*)d_integrity_check;
  T_CHECKS[0]=SZ(T_CHECKS); T_CHECKS[2]=(void*)d_context_check; T_CHECKS[3]=(void*)d_check_fn3; t21_init(); }
static const struct { const char*uuid; void**t; const char*name; } DARK[]={
  {"6bd5fb6c5bf4e74a8987d93912fd9df9",T_CUDART,"CUDART_INTERFACE"}, {"42d85a8123f6cb478298f6e78a3aecdc",T_TOOLS_TLS,"TOOLS_TLS"},
  {"a094798c2e742e7493f20800200c0a66",T_TOOLS_HOOKS,"TOOLS_RUNTIME_CALLBACK_HOOKS"}, {"c693336e1121df11a8c368f355d89593",T_CLS,"CONTEXT_LOCAL_STORAGE"},
  {"0ca50b8c1004929a89a7d0df10e77286",T_BYPASS,"CTX_CREATE_BYPASS"}, {"195bcbf4d67d024aacc51d29cea631ae",T_HEAP,"HEAP_ACCESS"},
  {"b10541e1f7c7c74a9f64f223be99f1e2",T_DEVEXT,"DEVICE_EXTENDED_RT"}, {"d4082055bde6704b8d34ba123c66e1f2",T_INTEG,"INTEGRITY_CHECK"},
  {"263e88607cd2614392f6bbd5006dfa7e",T_CHECKS,"CONTEXT_CHECKS"}, {"21318c60971432488ca641ff7324c8f2",T_21318,"CUBLASLT_CAPS(21318c60)"}, {0,0,0} };
CUresult cuGetExportTable(const void**t,const CUuuid*id){
  dark_init(); char hex[33]; const unsigned char*u=(const unsigned char*)id->b;
  for(int i=0;i<16;i++) sprintf(hex+2*i,"%02x",u[i]);
  for(int i=0;DARK[i].uuid;i++) if(!strcmp(DARK[i].uuid,hex)){ *t=DARK[i].t; if(trace) fprintf(stderr,"[warmd] export table %s\n",DARK[i].name); return OK; }
  // Unknown table: forward to the real driver library loaded locally WITHOUT cuInit (no context,
  // no GPU state). Works for host-only tables (e.g. the one cuBLASLt heuristics use).
  static void*real; static int(*rget)(const void**,const CUuuid*);
  if(getenv("SHIM_REAL_TABLES")){
    if(!real){ const char*p=getenv("SHIM_REAL_LIBCUDA"); real=dlopen(p?p:"/usr/lib/x86_64-linux-gnu/libcuda.so.570.211.01",RTLD_NOW|RTLD_LOCAL);
      if(real) rget=(int(*)(const void**,const CUuuid*))dlsym(real,"cuGetExportTable"); }
    if(rget){ int e=rget(t,id); if(trace) fprintf(stderr,"[warmd] export table {%s} -> forwarded to real driver: %d\n",hex,e); return e; } }
  if(trace) fprintf(stderr,"[warmd] cuGetExportTable {%s} unknown -> INVALID_VALUE\n",hex); *t=0; return E_INVALID_VALUE; }

// integrity check (cudart 12 verifies the driver with this keyed hash)
static const uint8_t MIX[256]={
0x29,0x2E,0x43,0xC9,0xA2,0xD8,0x7C,0x01,0x3D,0x36,0x54,0xA1,0xEC,0xF0,0x06,0x13,0x62,0xA7,0x05,0xF3,0xC0,0xC7,0x73,0x8C,0x98,0x93,0x2B,0xD9,0xBC,0x4C,
0x82,0xCA,0x1E,0x9B,0x57,0x3C,0xFD,0xD4,0xE0,0x16,0x67,0x42,0x6F,0x18,0x8A,0x17,0xE5,0x12,0xBE,0x4E,0xC4,0xD6,0xDA,0x9E,0xDE,0x49,0xA0,0xFB,0xF5,0x8E,
0xBB,0x2F,0xEE,0x7A,0xA9,0x68,0x79,0x91,0x15,0xB2,0x07,0x3F,0x94,0xC2,0x10,0x89,0x0B,0x22,0x5F,0x21,0x80,0x7F,0x5D,0x9A,0x5A,0x90,0x32,0x27,0x35,0x3E,
0xCC,0xE7,0xBF,0xF7,0x97,0x03,0xFF,0x19,0x30,0xB3,0x48,0xA5,0xB5,0xD1,0xD7,0x5E,0x92,0x2A,0xAC,0x56,0xAA,0xC6,0x4F,0xB8,0x38,0xD2,0x96,0xA4,0x7D,0xB6,
0x76,0xFC,0x6B,0xE2,0x9C,0x74,0x04,0xF1,0x45,0x9D,0x70,0x59,0x64,0x71,0x87,0x20,0x86,0x5B,0xCF,0x65,0xE6,0x2D,0xA8,0x02,0x1B,0x60,0x25,0xAD,0xAE,0xB0,
0xB9,0xF6,0x1C,0x46,0x61,0x69,0x34,0x40,0x7E,0x0F,0x55,0x47,0xA3,0x23,0xDD,0x51,0xAF,0x3A,0xC3,0x5C,0xF9,0xCE,0xBA,0xC5,0xEA,0x26,0x2C,0x53,0x0D,0x6E,
0x85,0x28,0x84,0x09,0xD3,0xDF,0xCD,0xF4,0x41,0x81,0x4D,0x52,0x6A,0xDC,0x37,0xC8,0x6C,0xC1,0xAB,0xFA,0x24,0xE1,0x7B,0x08,0x0C,0xBD,0xB1,0x4A,0x78,0x88,
0x95,0x8B,0xE3,0x63,0xE8,0x6D,0xE9,0xCB,0xD5,0xFE,0x3B,0x00,0x1D,0x39,0xF2,0xEF,0xB7,0x0E,0x66,0x58,0xD0,0xE4,0xA6,0x77,0x72,0xF8,0xEB,0x75,0x4B,0x0A,
0x31,0x44,0x50,0xB4,0x8F,0xED,0x1F,0x1A,0xDB,0x99,0x8D,0x33,0x9F,0x11,0x83,0x14};
static void ic_single(uint8_t*a,uint8_t x){
  uint8_t t1=a[0x40]; a[t1+0x10]=x; size_t t2=t1; uint8_t t3=(t1+1)&0xf;
  a[t1+0x20]=a[t2]^x; uint8_t t4=MIX[(uint8_t)(x^a[0x41])]; t1=a[t2+0x30]; a[t2+0x30]=t4^t1; a[0x41]=t4^t1; a[0x40]=t3;
  if(t3) return;
  uint8_t u=0x29, k=0;
  for(;;){ u^=a[0]; a[0]=u; for(int j=1;j<0x30;j++){ u=a[j]^MIX[u]; a[j]=u; }
    u=(uint8_t)(u+k); k++; if(k==0x12) break; u=MIX[u]; } }
static void ic_pass(uint8_t*a,const void*m,size_t n,uint8_t xm){ const uint8_t*b=m; for(size_t i=0;i<n;i++) ic_single(a,b[i]^xm); }
static void ic_pass5(uint8_t*a,uint64_t out[2]){
  uint8_t t=16-a[64]; for(int i=0;i<t;i++) ic_single(a,t);
  for(int i=0x30;i<0x40;i++) ic_single(a,a[i]);
  memcpy(out,a,16); }
static const uint8_t IC_P1[16]={0x14,0x6A,0xDD,0xAE,0x53,0xA9,0xA7,0x52,0xAA,0x08,0x41,0x36,0x0B,0xF5,0x5A,0x9F};
#pragma pack(push,1)
typedef struct { uint32_t drv,ver,proc,thr; const void*cudart_t,*integ_t,*fn; uint64_t secs; } ic_in_t;
typedef struct { CUuuid g; int32_t dom,bus,dev; } ic_dev_t;
#pragma pack(pop)
static CUresult d_integrity_check(uint32_t version,uint64_t secs,uint64_t*res){
  LK; if(ensure_init()) return E_NOT_INITIALIZED;
  if(version%10==0){ res[0]=0x3341181C03CB675Cull; res[1]=0x8ED383AA1F4CD1E8ull; return OK; }
  if(version%10==1){ res[0]=0x1841181C03CB675Cull; res[1]=0x8ED383AA1F4CD1E8ull; return OK; }
  uint8_t a[66]={0}; ic_pass(a,IC_P1,16,0x36);
  ic_in_t in={(uint32_t)I[0],version,(uint32_t)getpid(),(uint32_t)pthread_self(),T_CUDART,T_INTEG,T_INTEG[1],secs};
  ic_pass(a,&in,sizeof in,0);
  ic_dev_t dv; memcpy(&dv.g,IDATA+256,16); dv.dom=ATTR[50]; dv.bus=ATTR[33]; dv.dev=ATTR[34];   // PCI_DOMAIN_ID, PCI_BUS_ID, PCI_DEVICE_ID
  ic_pass(a,&dv,sizeof dv,0);
  uint64_t p51[2]; ic_pass5(a,p51);
  for(int i=0;i<16;i++) a[i]=0; for(int i=48;i<66;i++) a[i]=0;
  ic_pass(a,IC_P1,16,0x5c); ic_pass(a,p51,16,0); ic_pass5(a,res);
  return OK; }

