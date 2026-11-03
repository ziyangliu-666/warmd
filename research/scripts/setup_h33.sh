#!/usr/bin/env bash
# VM setup for version 2 of the nvproxy change: driver 570.195.03, NVIDIA runtime, gVisor built twice.
V=570.195.03; cd /root; export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq gcc make linux-headers-$(uname -r) python3-pip pkg-config >/dev/null 2>&1
systemctl stop docker 2>/dev/null; apt-get purge -y -qq 'nvidia-*' 'libnvidia-*' >/dev/null 2>&1; rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null
curl -sSL -C - --retry 5 -m 500 -o d.run https://us.download.nvidia.com/tesla/$V/NVIDIA-Linux-x86_64-$V.run; sh d.run --check | tail -1; sh d.run --silent --no-questions --no-opengl-files --no-x-check --kernel-module-type=proprietary >/dev/null 2>&1; nvidia-modprobe -u -c=0; nvidia-smi -pm 1 >/dev/null; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-ctk.gpg; echo "deb [signed-by=/usr/share/keyrings/nvidia-ctk.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/\$(ARCH) /" > /etc/apt/sources.list.d/nvidia-ctk.list; apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq nvidia-container-toolkit >/dev/null 2>&1; nvidia-ctk runtime configure --runtime=docker >/dev/null 2>&1
curl -sSL -m 300 https://go.dev/dl/go1.27.3.linux-amd64.tar.gz | tar xz -C /usr/local; export PATH=$PATH:/usr/local/go/bin GOFLAGS=-buildvcs=false
curl -sSL -m 200 https://github.com/google/gvisor/releases/download/release-20261026.0/gvisor-x86_64.tar.bz2 -o g.tbz; mkdir -p /opt/gv; tar xjf g.tbz -C /opt/gv
git clone -q --depth 1 --branch go https://github.com/google/gvisor gvgo
for v in 1 2; do mkdir -p /opt/gv$v/gvisor-bin; cp /opt/gv/gvisor-bin/* /opt/gv$v/gvisor-bin/; cd /root/gvgo; git checkout -q -- .; python3 /root/patch_v$v.py pkg/sentry/devices/nvproxy/frontend.go
  CGO_ENABLED=0 go build -o /opt/gv$v/runsc ./runsc 2>&1 | grep -v downloading | tail -2; CGO_ENABLED=0 go build -o /opt/gv$v/gvisor-bin/gvisor_sentry ./runsc/cmd/sentry 2>&1 | grep -v downloading | tail -2; ls -la /opt/gv$v/gvisor-bin/gvisor_sentry | cut -c20-; done; cd /root
python3 - <<'PY'
import json, os
p = "/etc/docker/daemon.json"; d = json.load(open(p)) if os.path.exists(p) else {}
for v in (1, 2):
    d.setdefault("runtimes", {})[f"runsc{v}"] = {"path": f"/opt/gv{v}/runsc", "runtimeArgs": ["--nvproxy=true", "--nvproxy-docker=true", "--host-uds=open"]}
json.dump(d, open(p, "w"), indent=1)
PY
systemctl restart docker; sleep 3; python3 -m pip install -q --no-deps --target /root/tr "triton==2.3.1" 2>/dev/null; H=$(dirname $(find /root/tr -name cuda.h | head -1)); mkdir -p /x; chmod 777 /x
gcc -O2 xc.c -o /x/xc -I$H -l:libcuda.so.1; gcc -O2 xc_neg2.c -o /x/xc_neg2 -I$H -l:libcuda.so.1; ls -la /x | cut -c20-; docker pull -q ubuntu:22.04 | tail -1; echo setup-done
