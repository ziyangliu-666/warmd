#!/usr/bin/env bash
# Separate two variables behind the wrong outputs: segment already present, and concurrency.
cd /workspace/exp; R=results; S=/tmp/warmd-exp/seg.sock; W=/workspace/warmd-seg; E="env WARMD_SOCKET=$S LD_LIBRARY_PATH=$W/build/lib"
start() { $W/build/warmd --socket $S --sessions 64 --standbys 0 > $R/warmd_h12.log 2>&1 & HP=$!; for i in $(seq 100); do grep -q ACTIVE $R/warmd_h12.log && break; sleep 0.1; done; }
: > $R/h12b.jsonl
start
$E timeout 100 python3 h12.py warmd_seg 16 1 2>>$R/h12.err | tee -a $R/h12b.jsonl
$E timeout 100 python3 h12.py warmd_seg 16 1 2>>$R/h12.err | tee -a $R/h12b.jsonl
$E timeout 100 python3 h12.py warmd_seg 64 8 2>>$R/h12.err | tee -a $R/h12b.jsonl
kill $HP; sleep 3; start
$E timeout 100 python3 h12.py warmd_seg 64 8 2>>$R/h12.err | tee -a $R/h12b.jsonl
$E timeout 100 python3 h12.py warmd_seg 64 2 2>>$R/h12.err | tee -a $R/h12b.jsonl
kill $HP
