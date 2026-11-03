#!/usr/bin/env bash
cd /workspace/exp; R=results; export CKPT=/workspace/exp/cuda-checkpoint
gcc -O2 -shared -fPIC vf.c -o libvf.so -I/usr/local/cuda/include -lcuda || exit 1
[ -x $CKPT ] || { curl -sSL -m 60 -o $CKPT https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint; chmod +x $CKPT; }
$CKPT --help 2>&1 | head -2
: > $R/h8.jsonl
python3 h8.py register 2>>$R/h8.err | tee -a $R/h8.jsonl
python3 h8.py scrub 2>>$R/h8.err | tee -a $R/h8.jsonl
python3 h6b.py broker > $R/broker.log 2>&1 & BP=$!
for i in $(seq 200); do grep -q ready $R/broker.log && break; sleep 0.1; done
for t in fork ckpt zoo; do timeout 200 python3 h8.py $t 2>>$R/h8.err | tee -a $R/h8.jsonl; done
kill -9 $BP; pkill -9 -f spawn_main; pkill -9 -f "h8.py"
grep "host tier" $R/broker.log | head -3
tail -n 3 $R/h8.err | cut -c1-300
