#!/usr/bin/env bash
# H35: what a parked vLLM process holds, then the Result 35 run with cuBLAS cleared, and with a larger budget.
cd /workspace; R=/workspace/results; mkdir -p $R exp
pip install -q "vllm==0.6.6" safetensors 2>&1 | grep -v -i "warn\|notice" | tail -1
pip install -q --force-reinstall --no-deps "transformers==4.47.1" "tokenizers==0.21.0" "huggingface-hub==0.36.2" 2>&1 | grep -v -i "warn\|notice" | tail -1
python3 -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen2.5-1.5B-Instruct", local_dir="/workspace/qwen15", allow_patterns=["*.json","*.safetensors","*.txt"]))' 2>&1 | tail -1
cd /workspace/warmd && make -s build/lib/libwarmd_segment.so && python3 tests/segment_tests.py 2>&1 | tail -1
curl -sSL -m 60 -o /workspace/exp/cuda-checkpoint https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint; chmod +x /workspace/exp/cuda-checkpoint
cd /workspace/exp; export F=/dev/shm/qwen15_vllm.safetensors SWAP=2
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader | tee $R/h35_env.txt
echo setup-done
clean() { ps -eo pid,args | grep -E "[w]armd.segments|[s]pawn_main|[p]ython3 h34.py" | awk '{print $1}' | xargs -r kill -9; sleep 5; rm -f $F.done; }
clean; timeout 900 python3 h34.py probe /workspace/exp/h34_trace.json 600 > $R/h35_probe.out 2> $R/h35_probe.err
grep '^{' $R/h35_probe.out | tee $R/h35_probe.jsonl
for run in "1 18" "1 24"; do set -- $run; clean; n=h35_clear$1_budget$2
  CLEAR=$1 BUDGET_GIB=$2 HIB_AFTER=2 ACTIVE=6 timeout 2400 python3 h34.py seg /workspace/exp/h34_trace.json 600 > $R/$n.out 2> $R/$n.err
  grep "^{" $R/$n.out | tail -1 | tee -a $R/h35.jsonl
  grep -E "Error|Traceback|had exited" $R/$n.err | grep -v EOFError | tail -n 3 | cut -c1-200
done
echo h35-done
