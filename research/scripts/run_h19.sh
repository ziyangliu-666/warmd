#!/usr/bin/env bash
# H19: run on a VM with Docker and the NVIDIA runtime. Each case uses one owner and one client.
D=/x; mkdir -p $D; chmod 777 $D; cp xc $D/; IMG=ubuntu:22.04; G="--gpus all"; MIB=1024
wait_ready() { for i in $(seq 100); do grep -q "owner ready" $1 && return 0; sleep 0.2; done; echo "  owner not ready"; cat $1; return 1; }
own() { docker rm -f xo >/dev/null 2>&1; docker run -d --name xo $G -v $D:$D $IMG $D/xc owner $D/s "$MIB" "$@" >/dev/null; for i in $(seq 100); do docker logs xo 2>&1 | grep -q "owner ready" && return 0; sleep 0.2; done; docker logs xo 2>&1 | tail -3; }
echo "== A: no containers"; ./xc owner /tmp/s $MIB > /tmp/o.log 2>&1 & P=$!; wait_ready /tmp/o.log; ./xc client /tmp/s $MIB; kill $P
echo "== B: two containers, separate pid, mount, net and ipc namespaces"; own; docker run --rm $G -v $D:$D $IMG $D/xc client $D/s $MIB; echo "  rc=$?"; docker run --rm $G -v $D:$D $IMG sh -c 'cat /proc/self/uid_map | head -1; readlink /proc/self/ns/pid'; readlink /proc/self/ns/pid
echo "== C: client with another uid"; docker run --rm $G --user 65534:65534 -v $D:$D $IMG $D/xc client $D/s $MIB; echo "  rc=$?"
echo "== D: owner container exits after sending"; docker rm -f xo >/dev/null; own exit; docker run --rm $G -v $D:$D $IMG $D/xc client $D/s $MIB; echo "  rc=$?"; docker ps -a --filter name=xo --format '  owner state: {{.Status}}'
echo "== E: client with no capabilities and read-only rootfs"; docker rm -f xo >/dev/null; own; docker run --rm $G --cap-drop=ALL --read-only --security-opt no-new-privileges -v $D:$D $IMG $D/xc client $D/s $MIB; echo "  rc=$?"
docker rm -f xo >/dev/null 2>&1
if command -v runsc >/dev/null; then echo "== F: owner in runc, client in gVisor with nvproxy"; own; docker run --rm --runtime=runsc $G -v $D:$D $IMG $D/xc client $D/s $MIB 2>&1 | tail -6 | cut -c1-220; echo "  rc=$?"; docker rm -f xo >/dev/null 2>&1; fi
