#!/usr/bin/env bash
# Build gVisor with the nvproxy patch, then repeat the failing case and the control.
cd /root; [ -x /usr/local/go/bin/go ] || { curl -sSL -m 200 https://go.dev/dl/go1.27.3.linux-amd64.tar.gz | tar xz -C /usr/local; }
export PATH=$PATH:/usr/local/go/bin GOFLAGS=-buildvcs=false; go version
cd gvgo; git checkout -q -- . ; python3 /root/gvisor_nvproxy_host_fd.py pkg/sentry/devices/nvproxy/frontend.go || exit 1; git diff --stat | tail -1
t=$(date +%s); CGO_ENABLED=0 go build -o /opt/gvp/runsc ./runsc 2>&1 | tail -5 | cut -c1-200; echo "build seconds: $(( $(date +%s) - t ))"; ls -la /opt/gvp/runsc | cut -c20-; /opt/gvp/runsc --version | head -1
cd /root; D=/x; IMG=ubuntu:22.04; MIB=1024
python3 - <<'PY'
import json
p = "/etc/docker/daemon.json"; d = json.load(open(p)); a = d["runtimes"]["runsc"]
d["runtimes"]["runscp"] = {"path": "/opt/gvp/runsc", "runtimeArgs": a["runtimeArgs"]}; json.dump(d, open(p, "w"), indent=1)
PY
systemctl restart docker; sleep 4; R="--runtime=runscp --gpus all -v $D:$D"
echo "== control, one sandbox, patched runsc"; docker run --rm $R $IMG sh -c "$D/xc owner /tmp/s $MIB & sleep 4; $D/xc client /tmp/s $MIB; echo client-rc=\$?" 2>&1 | tail -5 | cut -c1-200
echo "== F patched: owner in runc, client in gVisor"; docker rm -f xo >/dev/null 2>&1; docker run -d --name xo --gpus all -v $D:$D $IMG $D/xc owner $D/s $MIB >/dev/null; sleep 4; docker run --rm $R $IMG $D/xc client $D/s $MIB 2>&1 | tail -5 | cut -c1-200
echo "== F patched, second sandbox at once"; docker run --rm $R $IMG $D/xc client $D/s $MIB 2>&1 | tail -2 | cut -c1-200
echo "== negative: a non-control host fd is rejected"; docker rm -f xo >/dev/null 2>&1
