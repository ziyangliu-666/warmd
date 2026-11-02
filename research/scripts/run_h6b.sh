#!/usr/bin/env bash
# Device-memory file tests on PyTorch. The broker runs in its own process for every test.
cd /workspace/exp; R=results; export GB=8
gcc -O2 -shared -fPIC vf.c -o libvf.so -I/usr/local/cuda/include -lcuda || exit 1
python3 h6b.py native 2>>$R/h6b.err | tee $R/h6b.jsonl
python3 h6b.py broker >$R/broker.log 2>&1 & BP=$!
for i in $(seq 200); do grep -q ready $R/broker.log && break; sleep 0.1; done
for t in equal park fault share; do python3 h6b.py $t 2>>$R/h6b.err | tee -a $R/h6b.jsonl; done
kill $BP
tail -3 $R/h6b.err $R/broker.log | cut -c1-300
