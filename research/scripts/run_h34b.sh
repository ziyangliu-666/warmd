#!/usr/bin/env bash
# The three H34 runs, after setup. Stops the series at the first run that fails to start.
cd /workspace/exp; R=/workspace/results; : > $R/h34.jsonl
for run in "seg 600 x" "plain 600 x" "seg 300 150"; do set -- $run
  ps -eo pid,args | grep -E "[w]armd.segments|[s]pawn_main" | awk '{print $1}' | xargs -r kill -9; sleep 5
  rm -f /workspace/qwen15_vllm.safetensors.done
  KILL_AT=$([ $3 = x ] && echo 1e9 || echo $3) timeout 2400 python3 h34.py $1 /workspace/exp/h34_trace.json $2 > $R/h34_$1_$2_$3.out 2> $R/h34_$1_$2_$3.err
  grep "^{" $R/h34_$1_$2_$3.out | tail -1 | tee -a $R/h34.jsonl
  grep -E "Error|error|Traceback" $R/h34_$1_$2_$3.err | grep -v "^INFO\|^WARNING" | tail -n 3 | cut -c1-200
done
echo h34-done
