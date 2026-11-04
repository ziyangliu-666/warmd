"""warmd's Python package. ZygotePool runs short tasks in processes forked from warm ones (docs/python.md)."""
from .pool import ZygotePool, TaskError

__all__ = ["ZygotePool", "TaskError"]
__version__ = "0.1.0"
