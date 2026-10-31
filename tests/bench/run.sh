#!/usr/bin/env bash
# Benchmarks on a GPU machine. Starts its own warmd. Results -> results/bench-<date>/.
set -e; cd "$(dirname "$0")/../.."; make -s
R=tests/bench/results/bench-$(date +%Y%m%d-%H%M); mkdir -p $R
CUDA=${CUDA:-/usr/local/cuda}
gcc -O2 tests/bench/micro.c -o build/micro_native -I$CUDA/include -lcuda
gcc -O2 tests/bench/micro.c -o build/micro_shim -I$CUDA/include -Lbuild/lib -lcuda -Wl,-rpath,$PWD/build/lib
SOCK=/tmp/warmd-bench/gpu0.sock
build/warmd --socket $SOCK --sessions 64 --standbys 0 --metrics-port 9499 > $R/warmd.log 2>&1 & HP=$!
trap "kill $HP 2>/dev/null" EXIT
for i in $(seq 100); do grep -q ACTIVE $R/warmd.log && break; sleep 0.1; done
export WARMD_SOCKET=$SOCK
S="env LD_LIBRARY_PATH=$PWD/build/lib"
echo "== micro: rpc/launch latency"; { IMPL=native build/micro_native rpc; IMPL=warmd build/micro_shim rpc; } | tee $R/micro_rpc.jsonl
echo "== micro: copy bandwidth";    { IMPL=native build/micro_native copy; IMPL=warmd build/micro_shim copy; } | tee $R/micro_copy.jsonl
echo "== e2e: many short torch tasks, each its own process"
{
  python3 tests/bench/torch_e2e.py exec 8 1 native;      python3 tests/bench/torch_e2e.py exec 24 8 native
  $S python3 tests/bench/torch_e2e.py exec 8 1 warmd;  $S python3 tests/bench/torch_e2e.py exec 24 8 warmd
  python3 tests/bench/torch_e2e.py zygote 16 1 native;   python3 tests/bench/torch_e2e.py zygote 48 8 native
  $S python3 tests/bench/torch_e2e.py zygote 16 1 warmd; $S python3 tests/bench/torch_e2e.py zygote 96 8 warmd
  $S python3 tests/bench/torch_e2e.py warm 32 1 warmd; $S python3 tests/bench/torch_e2e.py warm 256 8 warmd
  for z in 1 2 4 8; do $S python3 tests/bench/pool_bench.py $z 512; done
} 2>/dev/null | tee $R/e2e.jsonl
echo "results in $R"
