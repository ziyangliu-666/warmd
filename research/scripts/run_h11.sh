#!/usr/bin/env bash
cd /workspace/exp; R=results; RATE=${RATE:-3}; SECS=${SECS:-300}
start_broker() { python3 h6b.py broker > $R/broker.log 2>&1 & BP=$!; for i in $(seq 200); do grep -q ready $R/broker.log && break; sleep 0.1; done; }
stop_all() { kill -9 $BP 2>/dev/null; pkill -9 -f spawn_main; pkill -9 -f "h[0-9]+.py"; sleep 3; }
timeout 200 python3 h10.py lease 2>>$R/h11.err | tee $R/h10_lease.jsonl
start_broker; timeout 150 python3 h8.py zoo 2>>$R/h11.err | tee $R/h8_zoo_aligned.jsonl; stop_all
: > $R/h11.jsonl
for a in -1 0 10; do start_broker; HIB_AFTER=$a python3 h11.py $RATE $SECS 2>>$R/h11.err | tee -a $R/h11.jsonl; stop_all; done
tail -n 4 $R/h11.err | cut -c1-300
