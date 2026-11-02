"""Device-memory files: mmap(2) semantics for GPU memory, using only public CUDA APIs.

A broker owns physical device memory, one allocation per file. A client maps a file read-only at
a virtual address that it keeps for its lifetime, so tensors stay valid across eviction. Clients
hold no driver shim and have their own context, which keeps faults in the context that caused them.
"""
import array, ctypes, json, mmap, os, socket, struct, threading
import torch

LIB = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libvf.so"))
GRAN = 2 << 20
U64 = ctypes.c_uint64

def ck(r, what):
    if r:
        raise RuntimeError(f"{what}: CUDA error {r}")

def init():
    """Create or bind the primary context on this thread through PyTorch."""
    torch.zeros(1, device="cuda"); torch.cuda.synchronize()

def round_up(n):
    return (n + GRAN - 1) & ~(GRAN - 1)

def reserve(n):
    va = U64(); ck(LIB.vf_reserve(ctypes.c_size_t(n), ctypes.byref(va)), "reserve"); return va.value

def map_ro(va, n, h, ro=True):
    ck(LIB.vf_map(U64(va), ctypes.c_size_t(n), U64(h), int(ro)), "map")

def unmap(va, n):
    ck(LIB.vf_unmap(U64(va), ctypes.c_size_t(n)), "unmap")

def import_fd(fd):
    h = U64(); ck(LIB.vf_import(fd, ctypes.byref(h)), "import"); return h.value

def host_register(p, n):
    return LIB.vf_register(ctypes.c_void_p(p), ctypes.c_size_t(n))

def release(h):
    ck(LIB.vf_release(U64(h)), "release")

# --- safetensors layout: the file on disk is already a flat image of the device-memory file.
DTYPES = {"F16": (torch.float16, 2, 16, 2), "F32": (torch.float32, 2, 32, 4), "I64": (torch.int64, 0, 64, 8)}

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h

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

# --- broker: owns physical memory; the protocol is one JSON line and optional fd per reply.
class Broker:
    def __init__(self, sock_path):
        self.files = {}; self.host = {}; self.lock = threading.Lock(); self.path = sock_path; init()

    def _host(self, path):
        """Return a pinned image of a file, with its own layout. Tensors start on 4 KiB boundaries:
        with safetensors' 8-byte alignment, ResNet output differed from ordinary weights by 6e-4,
        and at 64 bytes or more it is identical. Pinning the read-only page cache fails with error 1."""
        if path in self.host:
            return self.host[path]
        off, hdr = read_header(path); layout = {}; pos = 0
        for k, t in hdr.items():
            pos = (pos + 4095) & ~4095; n = t["data_offsets"][1] - t["data_offsets"][0]
            layout[k] = dict(t, data_offsets=[pos, pos + n]); pos += n
        q = ctypes.c_void_p(); ck(LIB.vf_pin(ctypes.byref(q), ctypes.c_size_t(pos)), "pin")
        fd = os.open(path, os.O_RDONLY); mm = mmap.mmap(fd, os.path.getsize(path), access=mmap.ACCESS_COPY); os.close(fd)
        view = (ctypes.c_char * len(mm)).from_buffer(mm); base = ctypes.addressof(view)
        for k, t in hdr.items():
            ctypes.memmove(q.value + layout[k]["data_offsets"][0], base + off + t["data_offsets"][0], t["data_offsets"][1] - t["data_offsets"][0])
        del view; mm.close(); self.host[path] = (q.value, pos, layout)
        return self.host[path]

    def load(self, name, path):
        addr, size, layout = self._host(path); n = round_up(size)
        h = U64(); ck(LIB.vf_create(ctypes.c_size_t(n), ctypes.byref(h)), "create")
        va = reserve(n); map_ro(va, n, h.value, ro=False)
        ck(LIB.vf_write(U64(va), ctypes.c_void_p(addr), ctypes.c_size_t(size)), "write")
        unmap(va, n)
        efd = ctypes.c_int(); ck(LIB.vf_export(h, ctypes.byref(efd)), "export")
        self.files[name] = dict(h=h.value, fd=efd.value, size=n, off=0, hdr=layout, path=path, gen=self.files.get(name, {}).get("gen", 0) + 1)

    def evict(self, name):
        f = self.files[name]; os.close(f["fd"]); release(f["h"]); f["h"] = None; f["fd"] = None

    def handle(self, c):
        init()   # A CUDA context is current per thread, and this is a new thread.
        rf = c.makefile("r")
        for line in rf:
            q = json.loads(line); name = q.get("name"); fd = None; rep = {}
            with self.lock:
                if q["op"] == "open":
                    f = self.files.get(name)
                    if f is None or f["h"] is None:
                        self.load(name, q.get("path") or f["path"]); f = self.files[name]
                    rep = dict(size=f["size"], off=f["off"], hdr=f["hdr"], gen=f["gen"]); fd = f["fd"]
                elif q["op"] == "evict":
                    self.evict(name)
            data = (json.dumps(rep) + "\n").encode()
            if fd is None:
                c.sendall(data)
            else:
                socket.send_fds(c, [data], [fd])

    def serve(self):
        if os.path.exists(self.path):
            os.unlink(self.path)
        s = socket.socket(socket.AF_UNIX); s.bind(self.path); s.listen(64); print("broker ready", flush=True)
        while True:
            c, _ = s.accept(); threading.Thread(target=self.handle, args=(c,), daemon=True).start()

class Client:
    def __init__(self, sock_path):
        self.s = socket.socket(socket.AF_UNIX); self.s.connect(sock_path); self.buf = b""; self.maps = {}

    def _call(self, want_fd, **q):
        self.s.sendall((json.dumps(q) + "\n").encode())
        data, fds, _, _ = socket.recv_fds(self.s, 1 << 24, 1) if want_fd else (self.s.recv(1 << 24), [], 0, 0)
        while not data.endswith(b"\n"):
            data += self.s.recv(1 << 24)
        return json.loads(data), (fds[0] if fds else None)

    def open(self, name, path=None):
        """Map a file. Tensors point into a reserved range, which stays valid across evict and remap."""
        rep, fd = self._call(True, op="open", name=name, path=path)
        va = reserve(rep["size"]); h = import_fd(fd); os.close(fd); map_ro(va, rep["size"], h)
        self.maps[name] = dict(va=va, size=rep["size"], h=h, mapped=True)
        out = {}
        for k, t in rep["hdr"].items():
            _, code, bits, _ = DTYPES[t["dtype"]]; out[k] = tensor_at(va + rep["off"] + t["data_offsets"][0], t["shape"], code, bits)
        return out

    def park(self, name):
        """Drop the mapping and this process's reference to physical memory. The tensors stay, unusable."""
        m = self.maps[name]; torch.cuda.synchronize(); unmap(m["va"], m["size"]); release(m["h"]); m["mapped"] = False
        torch.cuda.empty_cache()   # Scratch memory is the only mutable device state, and a parked process needs none.

    def resume(self, name, path=None):
        """Map the file again at the same address. The broker reloads it if it was evicted."""
        m = self.maps[name]; rep, fd = self._call(True, op="open", name=name, path=path)
        m["h"] = import_fd(fd); os.close(fd); map_ro(m["va"], m["size"], m["h"]); m["mapped"] = True

    def evict(self, name):
        self._call(False, op="evict", name=name)
