#!/usr/bin/env bash
# H30: checkpoint in runc and gVisor, with default connections and one connection, then vLLM.
cd /x/exp; mv h27_fixed.py h27.py; X=/x; IMG=python:3.11-slim; R=/root/h30.jsonl; : > $R; E="env PYTHONPATH=/x/py HOME=/tmp"; V="-v $X:$X -v $X/exp:/workspace/exp -v $X/m:/workspace/m -w /workspace/exp"
docker rm -f xb >/dev/null 2>&1; docker run -d --name xb --gpus all -e GB=2 $V $IMG $E python3 h6b.py broker >/dev/null; for i in $(seq 120); do docker logs xb 2>&1 | grep -q ready && break; sleep 1; done
timeout 100 docker run --rm --gpus all $V $IMG $E python3 h27.py ckpt runc 2>/root/h30a.err | tail -1 | cut -c1-360 | tee -a $R
timeout 100 docker run --rm --gpus all -e CUDA_DEVICE_MAX_CONNECTIONS=1 $V $IMG $E python3 h27.py ckpt runc_connections1 2>/root/h30b.err | tail -1 | cut -c1-360 | tee -a $R
timeout 100 docker run --rm --runtime=runscp --gpus all $V $IMG $E python3 h27.py ckpt gvisor 2>/root/h30c.err | tail -1 | cut -c1-360 | tee -a $R
docker rm -f xb >/dev/null 2>&1; tail -n 2 /root/h30a.err /root/h30c.err | cut -c1-200
bash run_h27v.sh 2>&1 | tail -n 8 | cut -c1-520; cp /root/h27.jsonl /root/h30_vllm.jsonl; echo h30-done
