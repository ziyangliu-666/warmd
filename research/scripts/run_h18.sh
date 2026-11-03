#!/usr/bin/env bash
cd /workspace/exp; R=results
/workspace/vl/bin/python h6b.py broker > $R/broker.log 2>&1 & BP=$!
for i in $(seq 200); do grep -q ready $R/broker.log && break; sleep 0.1; done
timeout 280 /workspace/vl/bin/python h18.py 2>$R/h18.err | tee $R/h18.jsonl
kill -9 $BP; pkill -9 -f h18.py
grep -v "^INFO\|^WARNING\|^$\|Loading safetensors\|rank0" $R/h18.err | tail -n 8 | cut -c1-240
