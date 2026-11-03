#!/usr/bin/env bash
# VM setup for H27: replace the driver with one that has checkpoint support and is on nvproxy's list,
# then install the NVIDIA runtime, patched gVisor, PyTorch for containers, and vLLM's model.
V=570.195.03; cd /root; export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq gcc make linux-headers-$(uname -r) python3-pip pkg-config libglvnd-dev >/dev/null 2>&1
systemctl stop docker nvidia-persistenced 2>/dev/null; pkill -9 nvidia 2>/dev/null; apt-get purge -y -qq 'nvidia-*' 'libnvidia-*' >/dev/null 2>&1; rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null
curl -sSL -m 200 -o d.run https://us.download.nvidia.com/tesla/$V/NVIDIA-Linux-x86_64-$V.run; ls -la d.run | cut -c20-; sh d.run --silent --no-questions --no-opengl-files --no-x-check --kernel-module-type=proprietary 2>&1 | tail -2 | cut -c1-180; tail -n 3 /var/log/nvidia-installer.log | cut -c1-160
nvidia-modprobe -u -c=0 2>/dev/null; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-ctk.gpg; echo "deb [signed-by=/usr/share/keyrings/nvidia-ctk.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/\$(ARCH) /" > /etc/apt/sources.list.d/nvidia-ctk.list; apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq nvidia-container-toolkit >/dev/null 2>&1; nvidia-ctk runtime configure --runtime=docker 2>&1 | tail -1 | cut -c1-100; systemctl restart docker; sleep 3; docker run --rm --gpus all ubuntu:22.04 nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | tail -1 | cut -c1-120
curl -sSL -m 200 https://github.com/google/gvisor/releases/download/release-20261026.0/gvisor-x86_64.tar.bz2 -o g.tbz; mkdir -p /opt/gv /opt/gvp/gvisor-bin; tar xjf g.tbz -C /opt/gv; cp /opt/gv/gvisor-bin/* /opt/gvp/gvisor-bin/; ln -sf /opt/gv/runsc /usr/local/bin/runsc; runsc nvproxy list-supported-drivers 2>/dev/null | grep -c "^$V"
curl -sSL -m 200 https://go.dev/dl/go1.27.3.linux-amd64.tar.gz | tar xz -C /usr/local; export PATH=$PATH:/usr/local/go/bin GOFLAGS=-buildvcs=false; git clone -q --depth 1 --branch go https://github.com/google/gvisor gvgo; cd gvgo; python3 /x/exp/gvisor_nvproxy_host_fd.py pkg/sentry/devices/nvproxy/frontend.go; CGO_ENABLED=0 go build -o /opt/gvp/runsc ./runsc 2>&1 | grep -v downloading | tail -2; CGO_ENABLED=0 go build -o /opt/gvp/gvisor-bin/gvisor_sentry ./runsc/cmd/sentry 2>&1 | grep -v downloading | tail -2; CGO_ENABLED=0 go build -o /opt/gvp/gvisor-bin/checkpointgofer ./runsc/checkpointgofer 2>&1 | grep -v downloading | tail -2; cd /root
python3 - <<'PY'
import json, os
p = "/etc/docker/daemon.json"; d = json.load(open(p)) if os.path.exists(p) else {}
d.setdefault("runtimes", {})["runscp"] = {"path": "/opt/gvp/runsc", "runtimeArgs": ["--nvproxy=true", "--nvproxy-docker=true", "--host-uds=open"]}; json.dump(d, open(p, "w"), indent=1)
PY
systemctl restart docker; sleep 3
X=/x; IMG=python:3.11-slim; docker pull -q $IMG | tail -1; docker run --rm -v $X:$X $IMG pip install -q --target /x/py "torch==2.5.1" "transformers<4.50" accelerate safetensors "numpy<2" huggingface_hub --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple 2>&1 | grep -v notice | tail -1 | cut -c1-140
python3 -m pip install -q --no-deps --target /root/tr triton 2>&1 | grep -v -i "warning\|notice" | tail -1; H=$(find /root/tr -name cuda.h | head -1); gcc -O2 -shared -fPIC /x/exp/vf.c -o /x/exp/libvf.so -I$(dirname $H) -l:libcuda.so.1 2>&1 | grep -i " error" | head -3
curl -sSL -m 60 -o /x/exp/cuda-checkpoint https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint; chmod +x /x/exp/cuda-checkpoint; /x/exp/cuda-checkpoint --help 2>&1 | sed -n 2p | cut -c1-90
mkdir -p /x/m/2; V2="-v $X:$X -v $X/exp:/workspace/exp -v $X/m:/workspace/m -w /workspace/exp"; docker run --rm --gpus all $V2 $IMG env PYTHONPATH=/x/py HOME=/tmp python3 make_models.py 2>&1 | grep -v -i warn | tail -1 | cut -c1-100
echo setup-done
