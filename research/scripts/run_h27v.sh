#!/usr/bin/env bash
# vLLM under runc and gVisor. Installs into /x/vl, on top of the same PyTorch.
cd /x/exp; X=/x; IMG=python:3.11-slim; R=/root/h27.jsonl; V="-v $X:$X -v $X/exp:/workspace/exp -v $X/m:/workspace/m -w /workspace/exp"
docker run --rm -v $X:$X $IMG pip install -q --target /x/vl "vllm==0.6.6" --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple 2>&1 | grep -v notice | tail -1 | cut -c1-160
docker run --rm -v $X:$X -e PYTHONPATH=/x/vl -e HOME=/tmp $IMG python3 -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen2.5-0.5B-Instruct", local_dir="/x/qwen", allow_patterns=["*.json","*.safetensors","*.txt"]))' 2>&1 | tail -1 | cut -c1-120
ENVV="env PYTHONPATH=/x/vl HOME=/tmp"; docker rm -f xb >/dev/null 2>&1; docker run -d --name xb --gpus all -e GB=2 $V $IMG $ENVV python3 h6b.py broker >/dev/null; for i in $(seq 120); do docker logs xb 2>&1 | grep -q ready && break; sleep 1; done
for pair in "runc:" "gvisor:--runtime=runscp"; do n=${pair%%:*}; a=${pair#*:}; timeout 280 docker run --rm $a --gpus all $V $IMG $ENVV python3 h27.py vllm $n 2>/root/h27_vllm_$n.err | tail -1 | cut -c1-500 | tee -a $R; grep -v "^INFO\|^WARNING\|^$\|Loading safetensors" /root/h27_vllm_$n.err | tail -n 2 | cut -c1-180; done
docker rm -f xb >/dev/null 2>&1; echo vllm-pass-done
