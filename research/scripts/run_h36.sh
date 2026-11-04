#!/usr/bin/env bash
# H36: segments against vLLM's own sleep mode, both on vLLM 0.8.5 with the V0 engine, plus plain for reference.
cd /workspace; R=/workspace/results; mkdir -p $R exp
pip install -q "vllm==0.8.5" safetensors 2>&1 | grep -v -i "warn\|notice" | tail -2
pip install -q "transformers==4.51.3" "tokenizers<0.22" "huggingface-hub<1.0" 2>&1 | grep -v -i "warn\|notice" | tail -2   # 5.x breaks vLLM 0.8.5
python3 -c "import vllm, transformers, torch; print(vllm.__version__, transformers.__version__, torch.__version__)" | tee $R/h36_env.txt
python3 -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen2.5-1.5B-Instruct", local_dir="/workspace/qwen15", allow_patterns=["*.json","*.safetensors","*.txt"]))' 2>&1 | tail -1
cd /workspace/warmd && make -s build/lib/libwarmd_segment.so && python3 tests/segment_tests.py 2>&1 | tail -1
curl -sSL -m 60 -o /workspace/exp/cuda-checkpoint https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint; chmod +x /workspace/exp/cuda-checkpoint
cd /workspace/exp; export F=/dev/shm/qwen15_vllm.safetensors SWAP=2 VLLM_USE_V1=0
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a $R/h36_env.txt
echo setup-done
clean() { ps -eo pid,args | grep -E "[w]armd.segments|[s]pawn_main|[p]ython3 h34.py" | awk '{print $1}' | xargs -r kill -9; sleep 5; rm -f $F.done; }
clean; PROBES=2 timeout 900 python3 h34.py probe /workspace/exp/h34_trace.json 600 > $R/h36_probe.out 2> $R/h36_probe.err
grep '^{' $R/h36_probe.out | tee $R/h36_probe.jsonl
echo probe-done
for run in "seg 6" "sleep 6" "plain 7"; do set -- $run; clean; n=h36_$1_$2
  HIB_AFTER=2 ACTIVE=$2 BUDGET_GIB=18 timeout 2400 python3 h34.py $1 /workspace/exp/h34_trace.json 600 > $R/$n.out 2> $R/$n.err
  grep "^{" $R/$n.out | tail -1 | tee -a $R/h36.jsonl
  grep -E "Error|Traceback|had exited" $R/$n.err | grep -v EOFError | tail -n 3 | cut -c1-200
done
echo h36-done
