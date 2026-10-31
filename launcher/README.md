# warmd (Python package)

`warmd.ZygotePool` runs short GPU tasks, each in its own process, forked from warm processes.
It needs a running `warmd` daemon and the warmd `libcuda.so.1` on `LD_LIBRARY_PATH`.
Documentation: docs/pool.md in the warmd repository.
