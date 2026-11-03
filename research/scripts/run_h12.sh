#!/usr/bin/env bash
cd /workspace/exp; R=results; S=/tmp/warmd-exp/seg.sock; W=/workspace/warmd-seg; : > $R/h12.jsonl
bash make_warmd_seg.sh
$W/build/warmd --socket $S --sessions 64 --standbys 0 > $R/warmd_h12.log 2>&1 & HP=$!
for i in $(seq 100); do grep -q ACTIVE $R/warmd_h12.log && break; sleep 0.1; done
E="env WARMD_SOCKET=$S LD_LIBRARY_PATH=$W/build/lib"
$E timeout 200 python3 h12.py warmd_copy 8 1 2>>$R/h12.err | tee -a $R/h12.jsonl
$E timeout 200 python3 h12.py warmd_copy 16 4 2>>$R/h12.err | tee -a $R/h12.jsonl
$E timeout 200 python3 h12.py warmd_seg 64 1 2>>$R/h12.err | tee -a $R/h12.jsonl
$E timeout 200 python3 h12.py warmd_seg 256 8 2>>$R/h12.err | tee -a $R/h12.jsonl
kill $HP; sleep 3
python3 h6b.py broker > $R/broker.log 2>&1 & BP=$!
for i in $(seq 200); do grep -q ready $R/broker.log && break; sleep 0.1; done
timeout 200 python3 h12.py native_vf 8 1 2>>$R/h12.err | tee -a $R/h12.jsonl
timeout 200 python3 h12.py native_vf 32 8 2>>$R/h12.err | tee -a $R/h12.jsonl
kill -9 $BP; pkill -9 -f spawn_main; tail -n 3 $R/h12.err | cut -c1-300
