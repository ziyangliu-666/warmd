#!/usr/bin/env bash
# H19 gVisor cases. Control: owner and client inside one sandbox. Test: owner outside, client inside.
# The driver is 535.183.01; nvproxy lists 535.183.06, so the version is forced. This is not a supported setup.
D=/x; IMG=ubuntu:22.04; MIB=1024
python3 - <<'PY'
import json, os
p = "/etc/docker/daemon.json"; d = json.load(open(p)) if os.path.exists(p) else {}
d.setdefault("runtimes", {})["runsc"] = {"path": "/opt/gv/runsc", "runtimeArgs": ["--nvproxy=true", "--nvproxy-docker=true", "--nvproxy-driver-version=535.183.06", "--host-uds=open", "--debug-log=/tmp/runsc/", "--strace=false"]}
json.dump(d, open(p, "w"), indent=1)
PY
systemctl restart docker; sleep 4; R="--runtime=runsc --gpus all -v $D:$D"
echo "== G0: nvidia-smi free baseline in sandbox"; docker run --rm $R $IMG sh -c 'ls /dev/nvidia* | tr "\n" " "; echo; dmesg | head -2' 2>&1 | tail -4 | cut -c1-200
echo "== G1: control, owner and client in one sandbox"; docker run --rm $R $IMG sh -c "$D/xc owner /tmp/s $MIB & sleep 4; $D/xc client /tmp/s $MIB; echo client-rc=\$?; kill %1" 2>&1 | tail -8 | cut -c1-200
echo "== F: owner in runc, client in gVisor"; docker rm -f xo >/dev/null 2>&1; docker run -d --name xo --gpus all -v $D:$D $IMG $D/xc owner $D/s $MIB >/dev/null; sleep 4; docker logs xo 2>&1 | tail -1 | cut -c1-120; docker run --rm $R $IMG $D/xc client $D/s $MIB 2>&1 | tail -6 | cut -c1-200; docker rm -f xo >/dev/null 2>&1
echo "== F2: owner in gVisor, client in runc"; docker run -d --name xo $R $IMG $D/xc owner $D/s2 $MIB >/dev/null; sleep 5; docker logs xo 2>&1 | tail -1 | cut -c1-120; ls -la $D/s2 2>&1 | cut -c1-70; docker run --rm --gpus all -v $D:$D $IMG $D/xc client $D/s2 $MIB 2>&1 | tail -6 | cut -c1-200; docker rm -f xo >/dev/null 2>&1
echo "== H: client in an unprivileged user namespace"; ./xc owner /tmp/s3 $MIB > /tmp/o3.log 2>&1 & P=$!; sleep 3; unshare -Ur ./xc client /tmp/s3 $MIB 2>&1 | tail -4 | cut -c1-200; kill $P
