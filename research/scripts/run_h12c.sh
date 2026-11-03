#!/usr/bin/env bash
cd /workspace/exp; R=results; S=/tmp/warmd-exp/seg.sock; W=/workspace/warmd-seg; E="env WARMD_SOCKET=$S LD_LIBRARY_PATH=$W/build/lib"
$W/build/warmd --socket $S --sessions 64 --standbys 0 > $R/warmd_h12.log 2>&1 & HP=$!; for i in $(seq 100); do grep -q ACTIVE $R/warmd_h12.log && break; sleep 0.1; done
: > $R/h12c.jsonl
for c in "16 1" "64 2" "256 4" "256 8" "256 16"; do $E timeout 100 python3 h12.py warmd_seg $c 2>>$R/h12.err | tee -a $R/h12c.jsonl; done
kill $HP
