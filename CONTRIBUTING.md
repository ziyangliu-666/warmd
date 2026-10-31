# Contributing

Run `make` and `tests/run.sh` before sending a change. Changes to `src/` must keep every test
passing, with PyTorch and driver API results equal to native CUDA. If a change affects speed,
include numbers from `tests/bench/run.sh` and the hardware they came from.

To add a driver function: implement it in `src/shim.c`; if it needs the GPU, add an operation to
`src/proto.h` (bump `WD_PROTO_VERSION` if a struct changes) and handle it in `src/worker.c`,
recording any resource it creates on the session. Add a check to `tests/driver_features.c` or
`tests/torch_conformance.py`.

To support a new driver branch: run `contrib/tools/gen_stubs.sh`, then `tests/run.sh` and
`contrib/tools/ic_check.py`. If cudart or cuBLAS fail at start-up, run the program natively under
`contrib/tools/etspy.sh` to see which undocumented table entries it calls, implement them in
`src/shim_dark.c`, and add the branch to `VALIDATED[]` in `src/shim.c`.

C is C11 without allocation on the call path. Python needs 3.9 and only the standard library.

Documentation is short and plain. Say what a thing does and give numbers with their setup. No marketing words, no em dashes, no emoji, sentence-case
headings. `contrib/tools/style-check.sh` points out common slips.
