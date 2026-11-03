#!/usr/bin/env bash
# H28: decompose context overhead, then repeat under NVIDIA MPS. Native first, because MPS changes behaviour.
cd /x/exp; X=/x; R=/root/h28.jsonl; : > $R; IMG=python:3.11-slim; H=$(find /root/tr -name cuda.h | head -1); gcc -O2 h28.c -o h28 -I$(dirname $H) -l:libcuda.so.1 2>&1 | grep -i " error"
smi() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
bare() { n=$1; shift; b=$(smi); HOLD=2 env "$@" ./h28 > /tmp/b.out & sleep 1.5; u=$(smi); wait; echo "{\"bare\":\"$n\",\"smi_delta_mib\":$((u-b)),\"self\":$(cat /tmp/b.out)}" | tee -a $R; sleep 1; }
rate() { n=$1; shift; t=$(date +%s.%N); for i in 1 2 3 4 5 6 7 8; do env "$@" ./h28 >/dev/null & done; wait; e=$(date +%s.%N); python3 -c "print('{\"parallel_start\":\"$n\",\"per_s\":%.2f}' % (8/($e-$t)))" | tee -a $R; }
sweep() { p=$1; shift; bare "$p default" A=1 "$@"; bare "$p connections=1" CUDA_DEVICE_MAX_CONNECTIONS=1 "$@"; bare "$p lazy" CUDA_MODULE_LOADING=LAZY "$@"; bare "$p eager" CUDA_MODULE_LOADING=EAGER "$@"; rate "$p default" A=1 "$@"; }
V="-v $X:$X -v $X/exp:/workspace/exp -v $X/m:/workspace/m -w /workspace/exp"; E="env PYTHONPATH=/x/py HOME=/tmp"
sweep native
for t in mem conc wild; do timeout 150 docker run --rm --gpus all $V $IMG $E python3 h28_py.py $t native 2>/root/h28_$t.err | tail -n 6 | cut -c1-420 | tee -a $R; done
timeout 100 docker run --rm --gpus all -e CUDA_DEVICE_MAX_CONNECTIONS=1 $V $IMG $E python3 h28_py.py mem native_connections1 2>/dev/null | tail -1 | cut -c1-400 | tee -a $R
mkdir -p /tmp/mps /tmp/mpsl; export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps CUDA_MPS_LOG_DIRECTORY=/tmp/mpsl; nvidia-cuda-mps-control -d; sleep 2; ./h28 > /dev/null; sleep 1; echo "{\"mps_server_alone_mib\":$(smi)}" | tee -a $R
sweep mps
M="--ipc=host -v /tmp/mps:/tmp/mps -e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps"
for t in mem conc wild; do timeout 150 docker run --rm --gpus all $M $V $IMG $E python3 h28_py.py $t mps 2>/root/h28_mps_$t.err | tail -n 6 | cut -c1-420 | tee -a $R; done
echo quit | nvidia-cuda-mps-control; sleep 2; smi; tail -n 2 /tmp/mpsl/server.log 2>/dev/null | cut -c1-160
