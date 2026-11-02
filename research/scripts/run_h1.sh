#!/usr/bin/env bash
# H1: cold-start stages, native vs warmd, warm page cache. One disk-cold pass per size at the end.
cd /workspace/exp; R=results; mkdir -p $R
python3 make_models.py 2>&1 | grep -v Warning
W=/workspace/warmd; SOCK=/tmp/warmd-exp/gpu0.sock
$W/build/warmd --socket $SOCK --sessions 64 --standbys 0 > $R/warmd_h1.log 2>&1 & HP=$!
for i in $(seq 100); do grep -q ACTIVE $R/warmd_h1.log && break; sleep 0.1; done
run_w() { WARMD_SOCKET=$SOCK LD_LIBRARY_PATH=$W/build/lib python3 cold_stage.py "$@" 2>>$R/h1.err; }
run_n() { python3 cold_stage.py "$@" 2>>$R/h1.err; }
for gb in 2 8 16; do
  cat /workspace/m/$gb/model.safetensors > /dev/null
  for rep in 1 2 3 4; do run_n $gb native warm; run_w $gb warmd warm; done
done | tee $R/h1.jsonl
for gb in 2 8 16; do run_n $gb native cold; run_w $gb warmd cold; done | tee -a $R/h1.jsonl
kill $HP
