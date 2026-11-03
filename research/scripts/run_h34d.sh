#!/usr/bin/env bash
# H34d on an A100 container: setup, then three runs that split the wake time and compare at equal memory.
#   seg   HIB_AFTER=2   the Result 33 policy, with each wake timed in parts
#   seg   HIB_AFTER=30  hibernate only models idle for 30 s; hot models stay parked
#   plain ACTIVE=7      plain with one more engine, about the device memory that seg used
cd /workspace; R=/workspace/results; mkdir -p $R exp
pip install -q "vllm==0.6.6" safetensors 2>&1 | grep -v -i "warn\|notice" | tail -1
pip install -q --force-reinstall --no-deps "transformers==4.47.1" "tokenizers==0.21.0" "huggingface-hub==0.36.2" 2>&1 | grep -v -i "warn\|notice" | tail -1
python3 -c "import vllm, transformers, torch; print(vllm.__version__, transformers.__version__, torch.__version__)"
python3 -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen2.5-1.5B-Instruct", local_dir="/workspace/qwen15", allow_patterns=["*.json","*.safetensors","*.txt"]))' 2>&1 | tail -1
cd /workspace/warmd && make -s build/lib/libwarmd_segment.so && python3 tests/segment_tests.py 2>&1 | tail -1
curl -sSL -m 60 -o /workspace/exp/cuda-checkpoint https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint; chmod +x /workspace/exp/cuda-checkpoint
cd /workspace/exp; nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader | tee $R/h34d_env.txt
echo setup-done
for run in "seg 2 6" "seg 30 6" "plain 2 7"; do set -- $run
  ps -eo pid,args | grep -E "[w]armd.segments|[s]pawn_main|[p]ython3 h34.py" | awk '{print $1}' | xargs -r kill -9; sleep 5
  rm -f /workspace/qwen15_vllm.safetensors.done
  n=h34d_$1_$2_$3
  HIB_AFTER=$2 ACTIVE=$3 timeout 2400 python3 h34.py $1 /workspace/exp/h34_trace.json 600 > $R/$n.out 2> $R/$n.err
  grep "^{" $R/$n.out | tail -1 | tee -a $R/h34d.jsonl
  grep -E "Error|Traceback|had exited" $R/$n.err | tail -n 3 | cut -c1-200
  grep -v -E "preempted|Loading safetensors|^$" $R/$n.err | tail -c 3000 > $R/$n.tail.txt
done
echo h34d-done
