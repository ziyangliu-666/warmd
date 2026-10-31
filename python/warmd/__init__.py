"""Run short GPU tasks in separate processes forked from warm processes. See docs/python.md."""
from .pool import ZygotePool, TaskError

__all__ = ["ZygotePool", "TaskError"]
__version__ = "0.1.0"
