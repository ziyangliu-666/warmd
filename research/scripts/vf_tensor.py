"""Wrap a device address as a PyTorch tensor through DLPack. Loads no CUDA library by itself."""
import ctypes
import torch

class _DLDevice(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int32), ("id", ctypes.c_int32)]
class _DLType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]
class _DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice), ("ndim", ctypes.c_int32), ("dtype", _DLType),
                ("shape", ctypes.POINTER(ctypes.c_int64)), ("strides", ctypes.c_void_p), ("byte_offset", ctypes.c_uint64)]
class _DLManaged(ctypes.Structure):
    pass
_DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManaged))
_DLManaged._fields_ = [("t", _DLTensor), ("ctx", ctypes.c_void_p), ("deleter", _DELETER)]
_keep = []
_no_delete = _DELETER(lambda p: None)
_capsule = ctypes.pythonapi.PyCapsule_New
_capsule.restype = ctypes.py_object
_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

def tensor_at(addr, shape, code, bits):
    """Wrap a device address as a tensor. The address outlives the tensor, so the deleter does nothing."""
    dims = (ctypes.c_int64 * len(shape))(*shape)
    m = _DLManaged(); m.t.data = addr; m.t.device = _DLDevice(2, 0); m.t.ndim = len(shape)
    m.t.dtype = _DLType(code, bits, 1); m.t.shape = dims; m.t.strides = None; m.t.byte_offset = 0
    m.ctx = None; m.deleter = _no_delete; _keep.append((m, dims))
    return torch.from_dlpack(_capsule(ctypes.addressof(m), b"dltensor", None))

