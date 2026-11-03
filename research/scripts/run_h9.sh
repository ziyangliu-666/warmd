#!/usr/bin/env bash
cd /workspace/exp; R=results; RATE=${RATE:-3}; SECS=${SECS:-300}
start_broker() { python3 h6b.py broker > $R/broker.log 2>&1 & BP=$!; for i in $(seq 200); do grep -q ready $R/broker.log && break; sleep 0.1; done; }
stop_all() { kill -9 $BP 2>/dev/null; pkill -9 -f spawn_main; pkill -9 -f "h[0-9].py"; sleep 3; }
: > $R/h9.jsonl
start_broker; timeout 150 python3 h8.py zoo 2>>$R/h9.err | tee $R/h8_zoo.jsonl; stop_all
for n in 1 4 16 32; do start_broker; timeout 300 python3 h9.py $n 2>>$R/h9.err | tee -a $R/h9.jsonl; stop_all; done
python3 h4.py native $RATE $SECS 2>>$R/h9.err | tee $R/h7b.jsonl
start_broker; python3 h7.py $RATE $SECS 2>>$R/h9.err | tee -a $R/h7b.jsonl; stop_all
start_broker; HIBERNATE=1 python3 h7.py $RATE $SECS 2>>$R/h9.err | tee -a $R/h7b.jsonl; stop_all
tail -n 4 $R/h9.err | cut -c1-300
