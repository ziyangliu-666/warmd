#!/usr/bin/env bash
# Run the full test suite on a machine with an NVIDIA GPU (driver >= 550) and PyTorch installed.
set -e; cd "$(dirname "$0")/.."
make -s
python3 tests/sys_tests.py "$@"
