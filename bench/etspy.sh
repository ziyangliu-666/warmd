#!/usr/bin/env bash
# Build the hidden-table spy as a proxy libcuda.so.1 in build/spy (real driver as absolute DT_NEEDED).
# Use (natively, no shim, no CUPTI):  ETSPY=all [ETSPY_DUMP=1] LD_LIBRARY_PATH=build/spy python3 task.py 2> spy.log
set -e; cd "$(dirname "$0")/.."; mkdir -p build/spy
REAL=$(readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so.1)
gcc -O2 -shared -fPIC tools/etspy.c -o build/spy/libcuda.so.1 -Wl,-soname,libcuda.so.1 -Wl,--no-as-needed $REAL -ldl
command -v patchelf >/dev/null || pip -q install patchelf
patchelf --replace-needed libcuda.so.1 $REAL build/spy/libcuda.so.1
echo "spy ready: build/spy/libcuda.so.1 -> $REAL"
