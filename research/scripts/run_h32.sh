#!/usr/bin/env bash
# H32 on an A100 container instance: context cost, start rate, stream cost, restore rate.
cd /workspace/exp; R=results; mkdir -p $R; : > $R/h32.jsonl; pip install -q "transformers<4.50" accelerate safetensors 2>&1 | grep -v -i "warn\|notice" | tail -1
nvidia-smi --query-gpu=name,driver_version,persistence_mode --format=csv,noheader | tee $R/h32_env.txt
gcc -O2 h28.c -o h28 -I/usr/local/cuda/include -lcuda || exit 1; gcc -O2 -shared -fPIC vf.c -o libvf.so -I/usr/local/cuda/include -lcuda || exit 1
smi() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
bare() { n=$1; shift; b=$(smi); HOLD=2 env "$@" ./h28 > /tmp/b.out & sleep 1.5; u=$(smi); wait; echo "{\"bare\":\"$n\",\"smi_delta_mib\":$((u-b)),\"self\":$(cat /tmp/b.out)}" | tee -a $R/h32.jsonl; sleep 1; }
rate() { n=$1; shift; t=$(date +%s.%N); for i in $(seq 16); do env "$@" ./h28 >/dev/null & done; wait; e=$(date +%s.%N); python3 -c "print('{\"parallel_16\":\"$n\",\"per_s\":%.2f}' % (16/($e-$t)))" | tee -a $R/h32.jsonl; }
for k in 1 2 3; do bare default A=1; bare connections1 CUDA_DEVICE_MAX_CONNECTIONS=1; rate default A=1; rate connections1 CUDA_DEVICE_MAX_CONNECTIONS=1; done
python3 h32.py 2>/dev/null | tee -a $R/h32.jsonl; CUDA_DEVICE_MAX_CONNECTIONS=1 python3 h32.py 2>/dev/null | tee -a $R/h32.jsonl
python3 make_models.py 2>&1 | grep -v -i warn | tail -1; curl -sSL -m 60 -o cuda-checkpoint https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint; chmod +x cuda-checkpoint
for c in default one; do python3 h6b.py broker > $R/broker.log 2>&1 & BP=$!; for i in $(seq 200); do grep -q ready $R/broker.log && break; sleep 0.1; done
  if [ $c = one ]; then export CUDA_DEVICE_MAX_CONNECTIONS=1; fi; timeout 200 python3 h9.py 8 2>$R/h32_rate.err | tail -1 | sed "s/^{/{\"connections\":\"$c\",/" | cut -c1-600 | tee -a $R/h32.jsonl; unset CUDA_DEVICE_MAX_CONNECTIONS; kill -9 $BP; pkill -9 -f spawn_main; sleep 3; done
echo h32-done
