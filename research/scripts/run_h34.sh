#!/usr/bin/env bash
# H34 on an A100 container: setup, the segment GPU tests, then three runs.
cd /workspace; R=/workspace/results; mkdir -p $R exp
pip install -q "vllm==0.6.6" safetensors 2>&1 | grep -v -i "warn\|notice" | tail -1
pip install -q --force-reinstall --no-deps "transformers==4.47.1" "tokenizers==0.21.0" 2>&1 | grep -v -i "warn\|notice" | tail -1
python3 -c "import vllm, transformers, torch; print(vllm.__version__, transformers.__version__, torch.__version__)"
python3 -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen2.5-1.5B-Instruct", local_dir="/workspace/qwen15", allow_patterns=["*.json","*.safetensors","*.txt"]))' 2>&1 | tail -1
cd /workspace/warmd && make -s build/lib/libwarmd_segment.so && python3 tests/segment_tests.py 2>&1 | tail -1 && timeout 400 python3 tests/segment_gpu_tests.py 2>&1 | tail -4 | tee $R/segment_gpu_tests.txt
curl -sSL -m 60 -o /workspace/exp/cuda-checkpoint https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint; chmod +x /workspace/exp/cuda-checkpoint
cd /workspace/exp; nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader | tee $R/h34_env.txt
for run in "seg 600 x" "plain 600 x" "seg 300 150"; do set -- $run; pkill -9 -f "warmd.segments|h34.py|spawn_main" 2>/dev/null; sleep 5
  KILL_AT=$([ $3 = x ] && echo 1e9 || echo $3) timeout 2400 python3 h34.py $1 /workspace/exp/h34_trace.json $2 > $R/h34_$1_$2.out 2> $R/h34_$1_$2.err; tail -1 $R/h34_$1_$2.out | tee -a $R/h34.jsonl; grep -v "^INFO\|^WARNING\|^$\|Loading safetensors\|rank0" $R/h34_$1_$2.err | tail -n 3 | cut -c1-200; done
echo h34-done
