#!/usr/bin/env bash
cd /workspace/exp; R=results; : > $R/h15.jsonl
timeout 120 python3 h15.py fd 2>>$R/h15.err | tee -a $R/h15.jsonl
python3 h6b.py broker > $R/broker.log 2>&1 & BP=$!
for i in $(seq 200); do grep -q ready $R/broker.log && break; sleep 0.1; done
timeout 200 python3 h15.py hf 2>>$R/h15.err | tee -a $R/h15.jsonl
kill -9 $BP; sleep 2
/workspace/vl/bin/python h6b.py broker > $R/broker.log 2>&1 & BP=$!
for i in $(seq 200); do grep -q ready $R/broker.log && break; sleep 0.1; done
timeout 280 /workspace/vl/bin/python h15.py vllm 2>>$R/h15.err | tee -a $R/h15.jsonl
kill -9 $BP; pkill -9 -f spawn_main; pkill -9 -f "h15.py"
grep -v "^INFO\|^WARNING\|^$" $R/h15.err | tail -n 6 | cut -c1-260
