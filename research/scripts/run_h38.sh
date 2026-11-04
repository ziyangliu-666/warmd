#!/usr/bin/env bash
# H38: the activation peak of a shared replica against max_num_batched_tokens, and what it costs.
cd /workspace; RES=/workspace/results; mkdir -p $RES exp
pip install -q "vllm==0.8.5" safetensors 2>&1 | grep -v -i "warn\|notice" | tail -2
pip install -q "transformers==4.51.3" "tokenizers<0.22" "huggingface-hub<1.0" 2>&1 | grep -v -i "warn\|notice" | tail -2
python3 -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen2.5-1.5B-Instruct", local_dir="/workspace/qwen15", allow_patterns=["*.json","*.safetensors","*.txt"]))' 2>&1 | tail -1
cd /workspace/warmd && make -s build/lib/libwarmd_segment.so && python3 tests/segment_tests.py 2>&1 | tail -1
cd /workspace/exp; export F=/dev/shm/qwen15_vllm.safetensors SWAP=2 VLLM_USE_V1=0 OMP_NUM_THREADS=2
echo setup-done
clean() { ps -eo pid,args | grep -E "[w]armd.segments|[s]pawn_main|[p]ython3 h37.py" | awk '{print $1}' | xargs -r kill -9; sleep 5; rm -f $F.done; }
for b in 2048 512 256; do clean
  R=7 MAXBT=$b SPEED=2 timeout 1800 python3 h37.py serve-shared /workspace/exp/h34_trace.json 300 > $RES/h38_$b.out 2> $RES/h38_$b.err
  grep '^{' $RES/h38_$b.out | tail -1 | tee -a $RES/h38.jsonl
done
echo h38-done
