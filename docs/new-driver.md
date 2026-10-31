# Validating a new driver branch

The shim is validated on the 570 branch. To add another branch:

1. Install the driver on a test machine and run `make`.
2. Regenerate the stub list from that driver with `tools/gen_stubs.sh`. Commit `shim/stubs.c` if
   it changed.
3. Run `tests/run.sh`. Every test must pass, and the PyTorch and driver API results must match
   native results.
4. Run `python3 tools/ic_check.py` to confirm the integrity hash matches the real driver.
5. If cudart or cuBLAS fail during start-up:
   - `WARMD_TRACE=1` prints each `cuGetExportTable` uuid and marks the unknown ones;
   - `bench/etspy.sh` builds the table spy. Run the failing program natively with
     `ETSPY=all LD_LIBRARY_PATH=build/spy` (add `ETSPY_DUMP=1` to dump memory) and implement the
     entries it uses in `shim/dark.c`. See [hidden-tables.md](hidden-tables.md).
6. Add the branch to `VALIDATED[]` in `shim/shim.c` and to [compatibility.md](compatibility.md).
