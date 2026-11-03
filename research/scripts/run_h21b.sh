#!/usr/bin/env bash
# Build the sidecars too; the Sentry holds the patched code. Scripts come from the matching release.
export PATH=$PATH:/usr/local/go/bin GOFLAGS=-buildvcs=false; mkdir -p /opt/gvp/gvisor-bin; cp /opt/gv/gvisor-bin/* /opt/gvp/gvisor-bin/
cd /root/gvgo; CGO_ENABLED=0 go build -o /opt/gvp/gvisor-bin/gvisor_sentry ./runsc/cmd/sentry 2>&1 | tail -3 | cut -c1-200; CGO_ENABLED=0 go build -o /opt/gvp/gvisor-bin/checkpointgofer ./runsc/checkpointgofer 2>&1 | tail -2 | cut -c1-200; CGO_ENABLED=0 go build -o /opt/gvp/gvisor-bin/runsc-metric-server ./runsc/cmd/metricserver 2>&1 | tail -2 | cut -c1-200; ls -la /opt/gvp/gvisor-bin/gvisor_sentry | cut -c20-
cd /root; H=$(find /root/tr -name cuda.h | head -1); gcc -O2 xc_neg.c -o /x/xc_neg -I$(dirname $H) -l:libcuda.so.1; D=/x; IMG=ubuntu:22.04; MIB=1024; R="--runtime=runscp --gpus all -v $D:$D"
echo "== control, one sandbox, patched"; docker run --rm $R $IMG sh -c "$D/xc owner /tmp/s $MIB & sleep 4; $D/xc client /tmp/s $MIB; echo client-rc=\$?" 2>&1 | tail -5 | cut -c1-200
echo "== F patched: owner in runc, client in gVisor"; docker rm -f xo >/dev/null 2>&1; docker run -d --name xo --gpus all -v $D:$D $IMG $D/xc owner $D/s $MIB >/dev/null; sleep 4; docker run --rm $R $IMG $D/xc client $D/s $MIB 2>&1 | tail -5 | cut -c1-200
echo "== F patched: a second sandbox"; docker run --rm $R $IMG $D/xc client $D/s $MIB 2>&1 | tail -2 | cut -c1-200
echo "== F unpatched, same setup"; docker run --rm --runtime=runsc --gpus all -v $D:$D $IMG $D/xc client $D/s $MIB 2>&1 | tail -2 | cut -c1-200; docker rm -f xo >/dev/null 2>&1
echo "== negative, patched"; docker run --rm $R $IMG $D/xc_neg 2>&1 | tail -2 | cut -c1-200
echo "== negative, runc"; docker run --rm --gpus all -v $D:$D $IMG $D/xc_neg 2>&1 | tail -2 | cut -c1-200
