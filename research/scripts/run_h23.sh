#!/usr/bin/env bash
X=/x; IMG=python:3.11-slim; PY="env PYTHONPATH=/x/py HOME=/tmp"; V="-v $X:$X -v $X/exp:/workspace/exp -v $X/m:/workspace/m -w /workspace/exp"
docker rm -f xb xby >/dev/null 2>&1; docker run -d --name xb --gpus all -e GB=2 $V $IMG $PY python3 h6b.py broker >/dev/null; for i in $(seq 120); do docker logs xb 2>&1 | grep -q ready && break; sleep 1; done
pair() { echo "== $1"; rm -rf $X/flags; mkdir -p $X/flags; docker rm -f xby >/dev/null 2>&1; docker run -d --name xby $3 --gpus all $V $IMG $PY python3 h23.py bystander >/dev/null; for i in $(seq 120); do [ -f $X/flags/ready ] && break; sleep 1; done; [ -f $X/flags/ready ] || { echo "  bystander not ready"; docker logs xby 2>&1 | tail -2 | cut -c1-200; }; timeout 90 docker run --rm $2 --gpus all $V $IMG $PY python3 h23.py writer 2>&1 | tail -1 | cut -c1-200; [ -f $X/flags/done ] || touch $X/flags/done; for i in $(seq 40); do [ "$(docker inspect -f '{{.State.Running}}' xby)" = false ] && break; sleep 1; done; docker logs xby 2>&1 | tail -1 | cut -c1-200; docker rm -f xby >/dev/null 2>&1; }
pair "writer runc, bystander runc" "" ""
pair "writer gVisor, bystander runc" "--runtime=runscp" ""
pair "writer runc, bystander gVisor" "" "--runtime=runscp"
pair "writer gVisor, bystander gVisor" "--runtime=runscp" "--runtime=runscp"
docker rm -f xb >/dev/null 2>&1
