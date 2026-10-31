import importlib, pickle, struct

_HDR = struct.Struct("!Q")


def send_frame(sock, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HDR.pack(len(data)) + data)


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        b = sock.recv(min(n - len(buf), 1 << 20))
        if not b:
            return None
        buf += b
    return bytes(buf)


def recv_frame(sock):
    h = _recv_exact(sock, _HDR.size)
    if h is None:
        return None
    (n,) = _HDR.unpack(h)
    data = _recv_exact(sock, n)
    return None if data is None else pickle.loads(data)


def resolve(ref):
    """'package.module:qualname' -> object"""
    mod, _, name = ref.partition(":")
    obj = importlib.import_module(mod)
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


# CUDA error classes. "lost": the shared context died (warmd failed over) - always safe to retry.
# "fault": a sticky device fault surfaced in this task - it may be the culprit or a bystander.
_LOST = ("busy or unavailable", "context lost", "DEVICES_UNAVAILABLE")
_FAULT = ("illegal memory access", "unspecified launch failure", "illegal instruction", "misaligned address",
          "hardware stack error", "invalid program counter", "ILLEGAL_ADDRESS", "LAUNCH_FAILED",
          "CUBLAS_STATUS_EXECUTION_FAILED", "CUDNN_STATUS_EXECUTION_FAILED")


def classify_error(e):
    s = str(e)
    if any(k in s for k in _LOST):
        return "lost"
    if any(k in s for k in _FAULT):
        return "fault"
    return "error"
