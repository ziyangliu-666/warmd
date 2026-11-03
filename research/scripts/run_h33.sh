#!/usr/bin/env bash
D=/x; IMG=ubuntu:22.04; MIB=1024
for v in 1 2; do R="--runtime=runsc$v --gpus all -v $D:$D"
  echo "== version $v: owner in runc, client in gVisor"; docker rm -f xo >/dev/null 2>&1; docker run -d --name xo --gpus all -v $D:$D $IMG $D/xc owner $D/s $MIB >/dev/null; sleep 4; docker run --rm $R $IMG $D/xc client $D/s $MIB 2>&1 | tail -3 | cut -c1-200; docker rm -f xo >/dev/null 2>&1
  echo "== version $v: control, one sandbox"; docker run --rm $R $IMG sh -c "$D/xc owner /tmp/s $MIB & sleep 4; $D/xc client /tmp/s $MIB" 2>&1 | tail -1 | cut -c1-200
  echo "== version $v: negative files"; docker run --rm $R $IMG $D/xc_neg2 2>&1 | tail -4 | cut -c1-120; done
echo "== runc negative files"; docker run --rm --gpus all -v $D:$D $IMG $D/xc_neg2 2>&1 | tail -4 | cut -c1-120; echo h33-done
