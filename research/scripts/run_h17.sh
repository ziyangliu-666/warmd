#!/usr/bin/env bash
# Replay one real 10-minute window under each design. Five of twenty models fit at once.
cd /workspace/exp; R=results; export TRACE=/workspace/exp/replay_d01.json; SECS=600
start_broker() { python3 h6b.py broker > $R/broker.log 2>&1 & BP=$!; for i in $(seq 200); do grep -q ready $R/broker.log && break; sleep 0.1; done; }
stop_all() { kill -9 $BP 2>/dev/null; pkill -9 -f spawn_main; pkill -9 -f "h[0-9]+.py"; sleep 3; }
python3 sim_cold.py 3 $SECS 3.88 0.011 | tee $R/h17_sim.jsonl
python3 h4.py native 3 $SECS 2>>$R/h17.err | tee $R/h17.jsonl
for a in -1 10 60; do start_broker; HIB_AFTER=$a python3 h11.py 3 $SECS 2>>$R/h17.err | tee -a $R/h17.jsonl; stop_all; done
