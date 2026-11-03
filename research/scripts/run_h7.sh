#!/usr/bin/env bash
# Replay one synthetic long-tail trace: in-process LRU (native, warmd), then one process per model.
cd /workspace/exp; R=results; W=/workspace/warmd; S=/tmp/warmd-exp/gpu0.sock; RATE=${RATE:-3}; SECS=${SECS:-300}
python3 h4.py native $RATE $SECS 2>>$R/h7.err | tee $R/h7.jsonl
$W/build/warmd --socket $S --sessions 64 --standbys 0 > $R/warmd_h7.log 2>&1 & HP=$!
for i in $(seq 100); do grep -q ACTIVE $R/warmd_h7.log && break; sleep 0.1; done
WARMD_SOCKET=$S LD_LIBRARY_PATH=$W/build/lib python3 h4.py warmd $RATE $SECS 2>>$R/h7.err | tee -a $R/h7.jsonl
kill $HP; sleep 2
python3 h6b.py broker > $R/broker.log 2>&1 & BP=$!
for i in $(seq 200); do grep -q ready $R/broker.log && break; sleep 0.1; done
python3 h7.py $RATE $SECS 2>>$R/h7.err | tee -a $R/h7.jsonl
kill $BP
