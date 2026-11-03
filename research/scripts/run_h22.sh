#!/usr/bin/env bash
# H22: real kernels across boundaries. Broker in one container; clients in a container and in gVisor.
X=/x; E=$X/exp; mkdir -p $E/results $X/m/2; cd /root; H=$(find /root/tr -name cuda.h | head -1)
gcc -O2 -shared -fPIC $E/vf.c -o $E/libvf.so -I$(dirname $H) -l:libcuda.so.1 2>&1 | grep -i " error" | head -3
IMG=python:3.11-slim; docker pull -q $IMG | tail -1
PY="env PYTHONPATH=/x/py HOME=/tmp"
[ -d $X/py/torch ] || docker run --rm -v $X:$X $IMG pip install -q --target /x/py "torch==2.5.1" "transformers<4.50" accelerate safetensors triton "numpy<2" --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple 2>&1 | tail -1 | cut -c1-160
V="-v $X:$X -v $X/exp:/workspace/exp -v $X/m:/workspace/m -w /workspace/exp"
docker run --rm --gpus all $V $IMG $PY python3 make_models.py 2>&1 | grep -v -i warn | tail -2 | cut -c1-120
SOCK=$X/vf.sock; sed -i "s#SOCK = \"/tmp/vf.sock\"#SOCK = \"$SOCK\"#" $E/h6b.py
docker rm -f xb >/dev/null 2>&1; docker run -d --name xb --gpus all $V $IMG $PY GB=2 python3 h6b.py broker >/dev/null; for i in $(seq 120); do docker logs xb 2>&1 | grep -q ready && break; sleep 1; done; docker logs xb 2>&1 | tail -1 | cut -c1-160
GB="-e GB=2"
echo "== native, runc"; docker run --rm --gpus all $GB $V $IMG $PY python3 h6b.py native 2>&1 | tail -1 | cut -c1-260
echo "== mapped, runc"; docker run --rm --gpus all $GB $V $IMG $PY python3 h6b.py equal 2>&1 | tail -1 | cut -c1-260
echo "== mapped, gVisor patched"; docker run --rm --runtime=runscp --gpus all $GB $V $IMG $PY python3 h6b.py equal 2>&1 | tail -1 | cut -c1-260
echo "== fault: bystander in runc, writer also in runc"; docker run --rm --gpus all $GB $V $IMG $PY python3 h6b.py fault 2>&1 | tail -1 | cut -c1-340
echo "== fault: all in gVisor patched"; docker run --rm --runtime=runscp --gpus all $GB $V $IMG $PY python3 h6b.py fault 2>&1 | tail -1 | cut -c1-340
docker rm -f xb >/dev/null 2>&1
