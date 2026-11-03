#!/usr/bin/env bash
cd /x/exp; X=/x; IMG=python:3.11-slim; R=/root/h27.jsonl; : > $R; ENVV="env PYTHONPATH=/x/py HOME=/tmp"; V="-v $X:$X -v $X/exp:/workspace/exp -v $X/m:/workspace/m -w /workspace/exp"
broker() { docker rm -f xb >/dev/null 2>&1; docker run -d --name xb --gpus all -e GB=2 $V $IMG $ENVV python3 h6b.py broker >/dev/null; for i in $(seq 120); do docker logs xb 2>&1 | grep -q ready && break; sleep 1; done; }
run() { t=$1; n=$2; a=$3; timeout ${4:-200} docker run --rm $a --gpus all $V $IMG $ENVV python3 h27.py $t $n 2>/root/h27_$t_$n.err | tail -1 | cut -c1-500 | tee -a $R; }
sed -i 's#SOCK = "/tmp/vf.sock"#SOCK = "/x/vf.sock"#' h6b.py
broker
for t in ladder ckpt fork; do run $t runc ""; run $t gvisor "--runtime=runscp"; done
docker rm -f xb >/dev/null 2>&1; echo first-pass-done
