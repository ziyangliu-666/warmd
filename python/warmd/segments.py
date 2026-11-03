"""Segments: read-only device memory that outlives the processes that use it. See docs/segments.md.

A broker owns one physical device allocation per segment and exports it as a file descriptor.
A client imports the descriptor and maps it read-only at a virtual address that it keeps for its
lifetime. The client uses the unmodified NVIDIA driver library and its own CUDA context, so a
fault stays in the process that caused it, and a store into a segment faults the writer.

Only names that the broker's operator registered can be opened. A client never passes a path:
the broker reads the file, and a client can read back what the broker loaded.

The layout and protocol code needs no GPU and no PyTorch. The device code needs libcuda and
build/lib/libwarmd_segment.so. The client needs PyTorch.
"""
import ctypes
import json
import os
import re
import socket
import struct
import threading

ALIGN = 4096           # tensor start alignment; at 8 bytes, cuDNN gave slightly different results
NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
MAX_LINE = 1 << 20

# safetensors dtype -> (DLPack type code, bits, bytes per element)
DTYPES = {"BF16": (4, 16, 2), "F16": (2, 16, 2), "F32": (2, 32, 4), "F64": (2, 64, 8),
          "I8": (0, 8, 1), "I16": (0, 16, 2), "I32": (0, 32, 4), "I64": (0, 64, 8),
          "U8": (1, 8, 1), "BOOL": (6, 8, 1)}


class SegmentError(Exception):
    pass


# ---------------------------------------------------------------- layout, no GPU

def read_header(path):
    """Return (data start, tensors) of a safetensors file, without the metadata entry."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise SegmentError(f"{path}: not a safetensors file")
        n = struct.unpack("<Q", raw)[0]
        if n > 100 << 20:
            raise SegmentError(f"{path}: header of {n} bytes is too large")
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return 8 + n, header


def plan(header, file_size, data_start):
    """Give each tensor a 4 KiB-aligned place in the segment. Return (layout, total bytes)."""
    layout, pos = {}, 0
    for name, t in header.items():
        if t.get("dtype") not in DTYPES:
            raise SegmentError(f"{name}: unsupported dtype {t.get('dtype')}")
        begin, end = t["data_offsets"]
        count = 1
        for d in t["shape"]:
            count *= d
        nbytes = count * DTYPES[t["dtype"]][2]
        if end - begin != nbytes or begin < 0 or data_start + end > file_size:
            raise SegmentError(f"{name}: offsets do not match shape and dtype")
        pos = (pos + ALIGN - 1) & ~(ALIGN - 1)
        layout[name] = dict(dtype=t["dtype"], shape=list(t["shape"]), offset=pos, nbytes=nbytes, source=data_start + begin)
        pos += nbytes
    return layout, pos


def round_up(n, granularity):
    return (n + granularity - 1) // granularity * granularity


# ---------------------------------------------------------------- protocol, no GPU
# One JSON line per request and per reply. A reply to "open" carries the segment's fd.

def send_message(sock, message, fd=None):
    data = (json.dumps(message) + "\n").encode()
    if fd is None:
        sock.sendall(data)
    else:
        socket.send_fds(sock, [data], [fd])


def recv_message(sock, buffer):
    """Read one line. Return (message, fd or None). `buffer` is a bytearray kept per connection."""
    fd = None
    while b"\n" not in buffer:
        data, fds, _, _ = socket.recv_fds(sock, 65536, 1)
        if not data:
            raise EOFError
        if fds:
            if fd is not None:
                os.close(fd)
            fd = fds[0]
        buffer += data
        if len(buffer) > MAX_LINE:
            raise SegmentError("message too long")
    line, _, rest = bytes(buffer).partition(b"\n")
    buffer[:] = rest
    return json.loads(line), fd


def parse_files(items):
    """Turn ["name=path", ...] into a registry."""
    out = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not NAME.match(name):
            raise SegmentError(f"bad segment {item!r}: use NAME=PATH, NAME of letters, digits, '.', '_', '-'")
        out[name] = os.path.abspath(path)
    return out


# ---------------------------------------------------------------- device layer

def _library():
    path = os.environ.get("WARMD_SEGMENT_LIB")
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [path] if path else [os.path.join(here, "libwarmd_segment.so"),
                                       os.path.join(here, "..", "..", "build", "lib", "libwarmd_segment.so"),
                                       "libwarmd_segment.so"]
    for c in candidates:
        try:
            return ctypes.CDLL(c)
        except OSError:
            continue
    raise SegmentError("libwarmd_segment.so not found: run make, or set WARMD_SEGMENT_LIB")


def _check(result, what):
    if result:
        raise SegmentError(f"{what}: CUDA error {result}")


class CudaDevice:
    """The broker's device operations. Each thread must call bind() before the others."""

    def __init__(self, ordinal=0):
        self.ordinal = ordinal
        self.cuda = ctypes.CDLL("libcuda.so.1")
        self.lib = _library()
        self.host = {}          # path -> (file identity, pinned address, size); weights are copied from here
        self.lock = threading.Lock()
        self.granularity = None

    def bind(self):
        """Make the device's primary context current on this thread."""
        d, ctx = ctypes.c_int(), ctypes.c_void_p()
        _check(self.cuda.cuInit(0), "cuInit")
        _check(self.cuda.cuDeviceGet(ctypes.byref(d), self.ordinal), "cuDeviceGet")
        _check(self.cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), d), "cuDevicePrimaryCtxRetain")
        _check(self.cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        if self.granularity is None:
            g = ctypes.c_size_t()
            _check(self.lib.seg_granularity(ctypes.byref(g)), "granularity")
            self.granularity = max(g.value, 2 << 20)

    def _host_image(self, path, layout, total):
        """Copy the file into pinned host memory, in segment layout. Copy again if the file changed."""
        st = os.stat(path)
        identity = (st.st_size, st.st_mtime_ns, st.st_ino)
        with self.lock:
            old = self.host.get(path)
            if old and old[0] == identity:
                return old[1], old[2]
            if old:
                self.lib.seg_host_free(ctypes.c_void_p(old[1]))
            p = ctypes.c_void_p()
            _check(self.lib.seg_host_alloc(ctypes.byref(p), ctypes.c_size_t(max(total, 1))), "host alloc")
            with open(path, "rb") as f:
                for t in layout.values():
                    f.seek(t["source"])
                    data = f.read(t["nbytes"])
                    ctypes.memmove(p.value + t["offset"], data, len(data))
            self.host[path] = (identity, p.value, total)
            return p.value, total

    def load(self, path, layout, total):
        """Create device memory, fill it, and return (handle, fd, size)."""
        addr, _ = self._host_image(path, layout, total)
        size = round_up(max(total, 1), self.granularity)
        h, va, fd = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_int()
        _check(self.lib.seg_create(ctypes.c_size_t(size), ctypes.byref(h)), "create")
        try:
            _check(self.lib.seg_reserve(ctypes.c_size_t(size), ctypes.c_size_t(self.granularity), ctypes.byref(va)), "reserve")
            _check(self.lib.seg_map(va, ctypes.c_size_t(size), h, 0), "map")
            _check(self.lib.seg_write(va, ctypes.c_void_p(addr), ctypes.c_size_t(total)), "write")
            _check(self.lib.seg_unmap(va, ctypes.c_size_t(size)), "unmap")
            _check(self.lib.seg_unreserve(va, ctypes.c_size_t(size)), "unreserve")
            _check(self.lib.seg_export(h, ctypes.byref(fd)), "export")
        except SegmentError:
            self.lib.seg_release(h)
            raise
        return h.value, fd.value, size

    def release(self, handle, fd):
        """Drop the broker's reference. Memory returns when no client maps it."""
        os.close(fd)
        _check(self.lib.seg_release(ctypes.c_uint64(handle)), "release")


# ---------------------------------------------------------------- broker

class Broker:
    """Serves registered segments over a Unix socket.

    allow_uids: uids that may open segments; None means the broker's own uid and root.
    admin_uids: uids that may evict; the broker's own uid and root.
    """

    def __init__(self, files, device=None, allow_uids=None):
        self.files = dict(files)
        self.device = device if device is not None else CudaDevice()
        me = os.getuid()
        self.allow = set(allow_uids) if allow_uids else {me, 0}
        self.admin = {me, 0}
        self.loaded = {}         # name -> dict(handle, fd, size, layout, generation)
        self.generation = {}
        self.mapped = {}         # name -> number of clients that map the loaded generation
        self.lock = threading.Lock()                          # guards the dicts only
        self.load_locks = {n: threading.Lock() for n in self.files}   # a slow load blocks its own name only

    def new_connection(self):
        return {}                # name -> times this connection mapped the loaded generation

    def end_connection(self, conn):
        with self.lock:
            for name, n in conn.items():
                if name in self.mapped:
                    self.mapped[name] = max(0, self.mapped[name] - n)
            conn.clear()

    def request(self, uid, req, conn=None):
        """Handle one request. Return (reply, fd or None). Client errors become replies.

        conn is the per-connection state from new_connection(). It counts mappings, so that
        evict can refuse while a client still maps a segment: a reload then would hold two copies.
        """
        conn = {} if conn is None else conn
        op, name = req.get("op"), req.get("name")
        if uid not in self.allow:
            return dict(ok=False, error="permission denied"), None
        if op == "list":
            with self.lock:
                return dict(ok=True, segments={n: dict(loaded=n in self.loaded, mapped=self.mapped.get(n, 0)) for n in self.files}), None
        if not isinstance(name, str) or name not in self.files:
            return dict(ok=False, error="unknown segment"), None
        if op == "open":
            with self.load_locks[name]:
                with self.lock:
                    s = self.loaded.get(name)
                if s is None:
                    try:
                        start, header = read_header(self.files[name])
                        layout, total = plan(header, os.path.getsize(self.files[name]), start)
                        handle, fd, size = self.device.load(self.files[name], layout, total)
                    except (OSError, ValueError, SegmentError) as e:
                        return dict(ok=False, error=f"load failed: {e}"), None
                    with self.lock:
                        self.generation[name] = self.generation.get(name, 0) + 1
                        s = self.loaded[name] = dict(handle=handle, fd=fd, size=size, layout=layout, generation=self.generation[name])
                        self.mapped[name] = 0
                with self.lock:
                    self.mapped[name] += 1
                    conn[name] = conn.get(name, 0) + 1
                public = {k: {f: v[f] for f in ("dtype", "shape", "offset", "nbytes")} for k, v in s["layout"].items()}
                return dict(ok=True, size=s["size"], layout=public, generation=s["generation"]), s["fd"]
        if op == "release":
            with self.lock:
                if conn.get(name, 0) > 0:
                    conn[name] -= 1
                    self.mapped[name] = max(0, self.mapped.get(name, 0) - 1)
            return dict(ok=True), None
        if op == "evict":
            if uid not in self.admin:
                return dict(ok=False, error="permission denied"), None
            with self.load_locks[name], self.lock:
                users = self.mapped.get(name, 0)
                if users and not req.get("force"):
                    return dict(ok=False, error=f"{users} clients still map {name}; park them first or pass force"), None
                s = self.loaded.pop(name, None)
                self.mapped.pop(name, None)
                if s is not None:
                    self.device.release(s["handle"], s["fd"])
            return dict(ok=True, evicted=s is not None), None
        return dict(ok=False, error=f"unknown op {op!r}"), None

    def _connection(self, conn):
        try:
            creds = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
            _, uid, _ = struct.unpack("3i", creds)
            self.device.bind()
            buffer, state = bytearray(), self.new_connection()
            try:
                while True:
                    req, stray = recv_message(conn, buffer)
                    if stray is not None:
                        os.close(stray)
                    reply, fd = self.request(uid, req if isinstance(req, dict) else {}, state)
                    send_message(conn, reply, fd)
            finally:
                self.end_connection(state)    # a client that exits maps nothing
        except (EOFError, ConnectionError, json.JSONDecodeError, SegmentError):
            pass
        finally:
            conn.close()

    def serve(self, path, mode=0o660):
        if os.path.exists(path):
            os.unlink(path)
        s = socket.socket(socket.AF_UNIX)
        s.bind(path)
        os.chmod(path, mode)
        s.listen(64)
        self.device.bind()
        while True:
            conn, _ = s.accept()
            threading.Thread(target=self._connection, args=(conn,), daemon=True).start()


# ---------------------------------------------------------------- client (needs PyTorch)

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
_DLManaged._fields_ = [("tensor", _DLTensor), ("ctx", ctypes.c_void_p), ("deleter", _DELETER)]
_NO_DELETE = _DELETER(lambda p: None)


def tensor_at(address, shape, dtype, device_index=0):
    """A PyTorch tensor over device memory that the caller keeps alive."""
    import torch
    code, bits, _ = DTYPES[dtype]
    dims = (ctypes.c_int64 * len(shape))(*shape)
    m = _DLManaged()
    m.tensor.data, m.tensor.device, m.tensor.ndim = address, _DLDevice(2, device_index), len(shape)
    m.tensor.dtype, m.tensor.shape, m.tensor.strides, m.tensor.byte_offset = _DLType(code, bits, 1), dims, None, 0
    m.ctx, m.deleter = None, _NO_DELETE
    new = ctypes.pythonapi.PyCapsule_New
    new.restype, new.argtypes = ctypes.py_object, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    t = torch.from_dlpack(new(ctypes.addressof(m), b"dltensor", None))
    t._warmd_keep = (m, dims)    # the struct must live as long as the tensor
    return t


class Client:
    """Maps segments into this process. Use from the thread that runs PyTorch on the device."""

    def __init__(self, path, device_index=0):
        import torch
        torch.zeros(1, device=f"cuda:{device_index}")     # make PyTorch's context current
        self.lib, self.device_index = _library(), device_index
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.connect(path)
        self.buffer = bytearray()
        self.maps = {}

    def _call(self, **req):
        send_message(self.sock, req)
        reply, fd = recv_message(self.sock, self.buffer)
        if not reply.get("ok"):
            if fd is not None:
                os.close(fd)
            raise SegmentError(reply.get("error", "request failed"))
        return reply, fd

    def _attach(self, name, va=None):
        reply, fd = self._call(op="open", name=name)
        size, h = reply["size"], ctypes.c_uint64()
        try:
            _check(self.lib.seg_import(fd, ctypes.byref(h)), "import")
        finally:
            os.close(fd)
        if va is None:
            v = ctypes.c_uint64()
            _check(self.lib.seg_reserve(ctypes.c_size_t(size), ctypes.c_size_t(2 << 20), ctypes.byref(v)), "reserve")
            va = v.value
        _check(self.lib.seg_map(ctypes.c_uint64(va), ctypes.c_size_t(size), h, 1), "map")
        return reply, va, h.value

    def open(self, name):
        """Map a segment. Return {tensor name: tensor}. The tensors are read-only."""
        if name in self.maps:
            raise SegmentError(f"{name} is already open")
        reply, va, h = self._attach(name)
        self.maps[name] = dict(va=va, size=reply["size"], handle=h, mapped=True, generation=reply["generation"], layout=reply["layout"])
        return {k: tensor_at(va + t["offset"], t["shape"], t["dtype"], self.device_index) for k, t in reply["layout"].items()}

    def park(self, name):
        """Unmap a segment. Its tensors keep their addresses and must not be used until resume()."""
        import torch
        m = self.maps[name]
        if not m["mapped"]:
            return
        torch.cuda.synchronize(self.device_index)
        _check(self.lib.seg_unmap(ctypes.c_uint64(m["va"]), ctypes.c_size_t(m["size"])), "unmap")
        _check(self.lib.seg_release(ctypes.c_uint64(m["handle"])), "release")
        m["mapped"] = False
        self._call(op="release", name=name)
        torch.cuda.empty_cache()     # scratch memory is the only mutable device state left

    def resume(self, name):
        """Map a parked segment again at the same address. The broker reloads it if it was evicted.

        Return True if the broker loaded a new generation, for example after the file changed.
        The tensors then hold the new weights. If the layout changed, resume fails and leaves the
        segment parked, because the tensors would point at the wrong bytes.
        """
        m = self.maps[name]
        if m["mapped"]:
            return False
        reply, fd = self._call(op="open", name=name)
        if reply["size"] != m["size"] or reply["layout"] != m["layout"]:
            os.close(fd)
            self._call(op="release", name=name)
            raise SegmentError(f"{name} changed layout; open it again")
        h = ctypes.c_uint64()
        try:
            _check(self.lib.seg_import(fd, ctypes.byref(h)), "import")
        finally:
            os.close(fd)
        _check(self.lib.seg_map(ctypes.c_uint64(m["va"]), ctypes.c_size_t(m["size"]), h, 1), "map")
        changed = reply["generation"] != m["generation"]
        m["handle"], m["mapped"], m["generation"] = h.value, True, reply["generation"]
        return changed

    def close(self, name):
        self.park(name)
        m = self.maps.pop(name)
        _check(self.lib.seg_unreserve(ctypes.c_uint64(m["va"]), ctypes.c_size_t(m["size"])), "unreserve")

    def list(self):
        return self._call(op="list")[0]["segments"]

    def evict(self, name, force=False):
        """Ask the broker to drop its copy. Needs the broker's uid or root. Refused while any
        client maps the segment, unless force: then mapped clients keep the old copy alive."""
        return self._call(op="evict", name=name, force=force)[0]["evicted"]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m warmd.segments", description="Serve read-only device-memory segments.")
    ap.add_argument("--socket", required=True)
    ap.add_argument("--file", action="append", default=[], metavar="NAME=PATH", help="register a safetensors file")
    ap.add_argument("--allow-uid", action="append", type=int, default=[], help="a uid that may open segments")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--mode", type=lambda s: int(s, 8), default=0o660, help="socket file mode, octal")
    a = ap.parse_args(argv)
    if not a.file:
        ap.error("register at least one --file")
    broker = Broker(parse_files(a.file), CudaDevice(a.device), a.allow_uid or None)
    print(f"serving {len(broker.files)} segments on {a.socket}", flush=True)
    broker.serve(a.socket, a.mode)


if __name__ == "__main__":
    main()
