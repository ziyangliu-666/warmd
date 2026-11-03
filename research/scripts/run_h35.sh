#!/usr/bin/env bash
# GPU experiments for H3 and H5, run serially so that timings do not interfere.
cd /workspace/exp; R=results; W=/workspace/warmd; SOCK=/tmp/warmd-exp/gpu0.sock
python3 -c 'import triton; print(triton.__version__)'
for m in overhead ipc fault; do python3 h3.py $m native 2>>$R/h3.err; done | tee $R/h3.jsonl
python3 h3.py wild native 2>>$R/h3.err | tee -a $R/h3.jsonl
python3 h3.py conc native 2>>$R/h3.err | tee -a $R/h3.jsonl
$W/build/warmd --socket $SOCK --sessions 64 --standbys 1 > $R/warmd_h3.log 2>&1 & HP=$!
for i in $(seq 100); do grep -q ACTIVE $R/warmd_h3.log && break; sleep 0.1; done
export WARMD_SOCKET=$SOCK
LD_LIBRARY_PATH=$W/build/lib python3 h3.py wild warmd 2>>$R/h3.err | tee -a $R/h3.jsonl
sleep 2
LD_LIBRARY_PATH=$W/build/lib python3 h3.py conc warmd 2>>$R/h3.err | tee -a $R/h3.jsonl
kill $HP; sleep 2
gcc -O2 h5.c -o h5 -I/usr/local/cuda/include -lcuda -lpthread 2>&1 | tail -3; ./h5 | tee $R/h5.jsonl
