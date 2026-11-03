#!/usr/bin/env bash
# H28b: control for the MPS result: persistence mode. Then the MPS cases that were cut off, and start rate.
cd /x/exp; X=/x; R=/root/h28b.jsonl; : > $R; IMG=python:3.11-slim
echo quit | CUDA_MPS_PIPE_DIRECTORY=/tmp/mps nvidia-cuda-mps-control 2>/dev/null; pkill -9 -f h28_py; pkill -f nvidia-cuda-mps; sleep 3; docker ps -q | xargs -r docker rm -f >/dev/null
smi() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
bare() { n=$1; shift; b=$(smi); HOLD=2 env "$@" ./h28 > /tmp/b.out & sleep 1.5; u=$(smi); wait; echo "{\"bare\":\"$n\",\"smi_delta_mib\":$((u-b)),\"self\":$(cat /tmp/b.out)}" | tee -a $R; sleep 1; }
rate() { n=$1; shift; t=$(date +%s.%N); for i in $(seq 16); do env "$@" ./h28 >/dev/null & done; wait; e=$(date +%s.%N); python3 -c "print('{\"parallel_16\":\"$n\",\"per_s\":%.2f}' % (16/($e-$t)))" | tee -a $R; }
four() { p=$1; shift; bare "$p default" A=1 "$@"; bare "$p connections=1" CUDA_DEVICE_MAX_CONNECTIONS=1 "$@"; rate "$p default" A=1 "$@"; rate "$p connections=1" CUDA_DEVICE_MAX_CONNECTIONS=1 "$@"; }
four "no_persistence"
nvidia-smi -pm 1 | tail -1 | cut -c1-80; sleep 2; four "persistence"
mkdir -p /tmp/mps /tmp/mpsl; export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps CUDA_MPS_LOG_DIRECTORY=/tmp/mpsl; nvidia-cuda-mps-control -d; sleep 2; ./h28 >/dev/null; four "persistence+mps"
V="-v $X:$X -v $X/exp:/workspace/exp -v $X/m:/workspace/m -w /workspace/exp"; E="env PYTHONPATH=/x/py HOME=/tmp"; M="--ipc=host -v /tmp/mps:/tmp/mps -e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps"
timeout 120 docker run --rm --gpus all $M $V $IMG $E python3 h28_py.py wild mps 2>/root/h28b_wild.err | tail -1 | cut -c1-360 | tee -a $R
timeout 110 docker run --rm --gpus all $M $V $IMG $E python3 h28_py.py conc mps 2>/root/h28b_conc.err | tail -n 4 | cut -c1-200 | tee -a $R
tail -n 2 /root/h28b_wild.err | cut -c1-200
echo quit | nvidia-cuda-mps-control; sleep 2; smi
