"""warmd: run many short GPU tasks, each in its own process, without paying CUDA init per task.

Pieces (see the repository README):
  * warmd           - resident CUDA host daemon, one per GPU
  * libcuda.so.1 shim  - put on LD_LIBRARY_PATH of task processes
  * warmd.ZygotePool - pre-warmed fork servers with fault quarantine (this package)
"""
from .pool import ZygotePool, TaskError

__all__ = ["ZygotePool", "TaskError"]
__version__ = "0.1.0"
