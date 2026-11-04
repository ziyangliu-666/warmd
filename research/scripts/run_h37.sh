#!/usr/bin/env bash
# H37: how many isolated replicas of one model fit with private weights and with one shared segment,
# how they serve one trace, and what a write into the shared weights does to the other replicas.
cd /workspace; RES=/workspace/results; mkdir -p $RES exp
pip install -q "vllm==0.8.5" safetensors 2>&1 | grep -v -i "warn\|notice" | tail -2
pip install -q "transformers==4.51.3" "tokenizers<0.22" "huggingface-hub<1.0" 2>&1 | grep -v -i "warn\|notice" | tail -2
python3 -c "import vllm, transformers, torch; print(vllm.__version__, transformers.__version__, torch.__version__)" | tee $RES/h37_env.txt
python3 -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen2.5-1.5B-Instruct", local_dir="/workspace/qwen15", allow_patterns=["*.json","*.safetensors","*.txt"]))' 2>&1 | tail -1
cd /workspace/warmd && make -s build/lib/libwarmd_segment.so && python3 tests/segment_tests.py 2>&1 | tail -1
cd /workspace/exp; export F=/dev/shm/qwen15_vllm.safetensors SWAP=2 VLLM_USE_V1=0 OMP_NUM_THREADS=2
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a $RES/h37_env.txt
echo setup-done
clean() { ps -eo pid,args | grep -E "[w]armd.segments|[s]pawn_main|[p]ython3 h37.py" | awk '{print $1}' | xargs -r kill -9; sleep 5; rm -f $F.done; }
for m in cap-private cap-shared; do clean
  timeout 1800 python3 h37.py $m > $RES/h37_$m.out 2> $RES/h37_$m.err; grep '^{' $RES/h37_$m.out | tee -a $RES/h37_cap.jsonl
done
FP=$(grep '"mode": "cap-private"' $RES/h37_cap.jsonl | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["fit"])')
FS=$(grep '"mode": "cap-shared"' $RES/h37_cap.jsonl | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["fit"])')
echo "fit private=$FP shared=$FS" | tee -a $RES/h37_env.txt
echo cap-done
# Serve with one replica fewer than fit, to leave room for vLLM's start-up profile.
for run in "serve-one 1" "serve-private $((FP-1))" "serve-shared $((FS-1))"; do set -- $run; clean
  R=$2 SPEED=2 FAULT_AT=$([ $1 = serve-shared ] && echo 150 || echo 1e9) timeout 1800 python3 h37.py $1 /workspace/exp/h34_trace.json 300 > $RES/h37_$1.out 2> $RES/h37_$1.err
  grep '^{' $RES/h37_$1.out | tail -1 | tee -a $RES/h37.jsonl
done
echo h37-done
