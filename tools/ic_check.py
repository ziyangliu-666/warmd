"""Check the shim's integrity-check hash against the real driver.
Extracts the hash code from shim/dark.c, calls the real driver's INTEGRITY_CHECK[1] with the same
inputs, and compares.   python3 tools/ic_check.py   (needs a GPU and the CUDA headers)"""
import os, subprocess, sys, tempfile
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = open(os.path.join(root, "shim/dark.c")).read()
a = s.index("static const uint8_t MIX[256]"); b = s.rindex("static CUresult d_integrity_check(uint32_t version")
code = r'''#define _GNU_SOURCE
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <cuda.h>
''' + s[a:b] + r'''
static void hexu(const char*s,CUuuid*u){ for(int i=0;i<16;i++) sscanf(s+2*i,"%2hhx",(unsigned char*)&u->bytes[i]); }
int main(){
  cuInit(0); CUuuid u; const void*ti=0,*tc=0;
  hexu("d4082055bde6704b8d34ba123c66e1f2",&u); cuGetExportTable(&ti,&u);
  hexu("6bd5fb6c5bf4e74a8987d93912fd9df9",&u); cuGetExportTable(&tc,&u);
  int drv; cuDriverGetVersion(&drv); CUdevice d; cuDeviceGet(&d,0); CUuuid g; cuDeviceGetUuid(&g,d);
  int dom,bus,dev; cuDeviceGetAttribute(&dom,CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,d); cuDeviceGetAttribute(&bus,CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,d); cuDeviceGetAttribute(&dev,CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,d);
  CUresult(*ic)(uint32_t,uint64_t,uint64_t*)=((void**)ti)[1]; int bad=0;
  uint32_t vers[]={12022,12043,12045,12082,12093};
  for(int k=0;k<5;k++){ uint32_t ver=vers[k]; uint64_t secs=1700000000, real[2]={0}; ic(ver,secs,real);
    uint8_t a[66]={0}; ic_pass(a,IC_P1,16,0x36);
    ic_in_t in={(uint32_t)drv,ver,(uint32_t)getpid(),(uint32_t)pthread_self(),tc,ti,((void**)ti)[1],secs}; ic_pass(a,&in,sizeof in,0);
    ic_dev_t dv; memcpy(&dv.g,&g,16); dv.dom=dom; dv.bus=bus; dv.dev=dev; ic_pass(a,&dv,sizeof dv,0);
    uint64_t p51[2],mine[2]; ic_pass5(a,p51); for(int i=0;i<16;i++) a[i]=0; for(int i=48;i<66;i++) a[i]=0;
    ic_pass(a,IC_P1,16,0x5c); ic_pass(a,p51,16,0); ic_pass5(a,mine);
    int ok=real[0]==mine[0]&&real[1]==mine[1]; bad+=!ok; printf("cudart %u: %s\n",ver,ok?"MATCH":"DIFF"); }
  return bad; }
'''
d = tempfile.mkdtemp(); open(d + "/ic.c", "w").write(code)
cuda = os.environ.get("CUDA", "/usr/local/cuda")
subprocess.check_call(["gcc", "-O1", d + "/ic.c", "-o", d + "/ic", f"-I{cuda}/include", "-lcuda", "-lpthread"])
sys.exit(subprocess.call([d + "/ic"]))
