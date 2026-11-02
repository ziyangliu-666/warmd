#!/usr/bin/env bash
# H2 in a software-emulated VM: a server outside runsc, and a client inside it.
cd /workspace/vm; E=/workspace/exp
gcc -O2 -static $E/peer_server.c -o peer_server; gcc -O2 -static $E/peer_client.c -o peer_client
[ -f k ] || ssh-keygen -q -t ed25519 -N '' -f k
cat > user-data <<UD
#cloud-config
users:
  - name: root
    ssh_authorized_keys: ["$(cat k.pub)"]
disable_root: false
ssh_pwauth: false
UD
echo "instance-id: h2" > meta-data; cloud-localds seed.img user-data meta-data
rm -f vm.qcow2; qemu-img create -q -f qcow2 -b base.img -F qcow2 vm.qcow2 8G
nohup qemu-system-x86_64 -accel tcg,thread=multi -smp 4 -m 4096 -nographic -drive file=vm.qcow2,if=virtio -drive file=seed.img,if=virtio,format=raw -nic user,model=virtio-net-pci,hostfwd=tcp:127.0.0.1:2231-:22 > vm.log 2>&1 &
S="ssh -p 2231 -i k -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 root@127.0.0.1"
for i in $(seq 30); do $S true 2>/dev/null && break; sleep 10; done
$S uname -r || { echo VM did not boot; tail -5 vm.log; exit 1; }
scp -P 2231 -i k -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null peer_server peer_client /tmp/runsc root@127.0.0.1:/root/ 2>/dev/null
scp -r -P 2231 -i k -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $(dirname $(find /tmp -maxdepth 4 -name gvisor_sentry -type f | head -1)) root@127.0.0.1:/root/gvisor-bin 2>/dev/null
$S 'cd /root; chmod +x runsc; mkdir -p /tmp/s;
 echo "== baseline, no sandbox"; ./peer_server /tmp/s/a.sock > srv0.log & sleep 1; timeout 120 ./peer_client /tmp/s/a.sock; cat srv0.log; pkill peer_server
 for mode in "" "--host-uds=open"; do for plat in systrap; do echo "== runsc $mode platform=$plat"; ./peer_server /tmp/s/a.sock > srv.log & sleep 1;
   timeout 300 ./runsc --network=none --platform=$plat $mode do /root/peer_client /tmp/s/a.sock 2>&1 | tail -8 | cut -c1-200; cat srv.log; pkill peer_server; sleep 1; done; done; ./runsc --version | head -1'
pkill qemu-system
