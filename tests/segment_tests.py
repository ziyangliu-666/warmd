"""Tests for warmd.segments that need no GPU: layout, protocol and broker policy.

    python3 tests/segment_tests.py

tests/segment_gpu_tests.py covers the device path on a machine with a GPU.
"""
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from warmd import segments as S


def write_safetensors(path, tensors):
    """tensors: {name: (dtype, shape, bytes)}. Writes a valid file with packed data."""
    header, blob = {}, b""
    for name, (dtype, shape, data) in tensors.items():
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[len(blob), len(blob) + len(data)])
        blob += data
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + blob)


class FakeDevice:
    """Records loads and releases instead of touching a GPU. The fd is the read end of a pipe."""

    def __init__(self):
        self.loads, self.releases = [], []

    def bind(self):
        pass

    def load(self, path, layout, total):
        r, w = os.pipe()
        os.close(w)
        self.loads.append((path, total))
        return len(self.loads), r, S.round_up(max(total, 1), 2 << 20)

    def release(self, handle, fd):
        os.close(fd)
        self.releases.append(handle)


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "m.safetensors")
        write_safetensors(self.path, {"a": ("F16", [3], b"\1\0\2\0\3\0"), "b": ("F32", [2, 2], bytes(range(16))), "c": ("BF16", [1], b"\7\7")})

    def test_offsets_are_aligned_and_sizes_match(self):
        start, header = S.read_header(self.path)
        layout, total = S.plan(header, os.path.getsize(self.path), start)
        for t in layout.values():
            self.assertEqual(t["offset"] % S.ALIGN, 0)
        self.assertEqual(layout["b"]["nbytes"], 16)
        self.assertEqual(total, layout["c"]["offset"] + 2)
        with open(self.path, "rb") as f:
            f.seek(layout["b"]["source"])
            self.assertEqual(f.read(16), bytes(range(16)))

    def test_rejects_size_mismatch(self):
        start, header = S.read_header(self.path)
        header["b"]["shape"] = [3, 2]
        with self.assertRaises(S.SegmentError):
            S.plan(header, os.path.getsize(self.path), start)

    def test_rejects_offsets_past_the_file(self):
        start, header = S.read_header(self.path)
        header["c"]["data_offsets"] = [1000, 1002]
        with self.assertRaises(S.SegmentError):
            S.plan(header, os.path.getsize(self.path), start)

    def test_rejects_unknown_dtype(self):
        start, header = S.read_header(self.path)
        header["a"]["dtype"] = "F8_E4M3"
        with self.assertRaises(S.SegmentError):
            S.plan(header, os.path.getsize(self.path), start)

    def test_rejects_short_file(self):
        p = os.path.join(self.dir, "short")
        open(p, "wb").write(b"abc")
        with self.assertRaises(S.SegmentError):
            S.read_header(p)


class ProtocolTests(unittest.TestCase):
    def test_message_and_fd_round_trip(self):
        a, b = socket.socketpair()
        r, w = os.pipe()
        S.send_message(a, {"x": 1}, r)
        S.send_message(a, {"y": 2})
        buf = bytearray()
        m1, fd1 = S.recv_message(b, buf)
        m2, fd2 = S.recv_message(b, buf)
        self.assertEqual((m1, m2), ({"x": 1}, {"y": 2}))
        self.assertIsNotNone(fd1)
        self.assertIsNone(fd2)
        os.write(w, b"z")
        self.assertEqual(os.read(fd1, 1), b"z")
        for fd in (r, w, fd1):
            os.close(fd)
        a.close()
        b.close()

    def test_parse_files(self):
        self.assertEqual(S.parse_files(["m=/tmp/x"]), {"m": "/tmp/x"})
        for bad in ("m", "=x", "a/b=x", "../m=x"):
            with self.assertRaises(S.SegmentError):
                S.parse_files([bad])


class BrokerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "m.safetensors")
        write_safetensors(self.path, {"w": ("F32", [4], bytes(16))})
        self.dev = FakeDevice()
        self.me = os.getuid()
        self.b = S.Broker({"m": self.path}, self.dev, allow_uids=[self.me, 4242])

    def test_open_returns_fd_and_layout(self):
        reply, fd = self.b.request(self.me, {"op": "open", "name": "m"})
        self.assertTrue(reply["ok"])
        self.assertIsNotNone(fd)
        self.assertEqual(reply["layout"]["w"]["nbytes"], 16)
        self.assertNotIn("source", reply["layout"]["w"])

    def test_second_open_reuses_the_load(self):
        self.b.request(self.me, {"op": "open", "name": "m"})
        self.b.request(4242, {"op": "open", "name": "m"})
        self.assertEqual(len(self.dev.loads), 1)

    def test_a_path_is_never_loaded(self):
        # A client that names a file must not make the broker read it.
        secret = os.path.join(self.dir, "secret.safetensors")
        write_safetensors(secret, {"s": ("U8", [1], b"!")})
        for req in ({"op": "open", "name": secret}, {"op": "open", "name": "m", "path": secret}, {"op": "open", "path": secret}):
            self.b.request(self.me, req)
        self.assertEqual([p for p, _ in self.dev.loads], [self.path])

    def test_unknown_name_and_op(self):
        self.assertFalse(self.b.request(self.me, {"op": "open", "name": "x"})[0]["ok"])
        self.assertFalse(self.b.request(self.me, {"op": "frob", "name": "m"})[0]["ok"])
        self.assertFalse(self.b.request(self.me, {})[0]["ok"])

    def test_uid_not_allowed(self):
        reply, fd = self.b.request(777, {"op": "open", "name": "m"})
        self.assertEqual((reply["ok"], fd), (False, None))
        self.assertEqual(self.dev.loads, [])

    def test_only_admin_evicts(self):
        self.b.request(self.me, {"op": "open", "name": "m"})
        self.assertFalse(self.b.request(4242, {"op": "evict", "name": "m"})[0]["ok"])
        self.assertTrue(self.b.request(self.me, {"op": "evict", "name": "m"})[0]["evicted"])
        self.assertEqual(self.dev.releases, [1])

    def test_open_after_evict_reloads_with_new_generation(self):
        g1 = self.b.request(self.me, {"op": "open", "name": "m"})[0]["generation"]
        self.b.request(self.me, {"op": "evict", "name": "m"})
        g2 = self.b.request(self.me, {"op": "open", "name": "m"})[0]["generation"]
        self.assertEqual((g1, g2, len(self.dev.loads)), (1, 2, 2))

    def test_broken_file_is_an_error_not_a_crash(self):
        open(self.path, "wb").write(b"\xff" * 4)
        reply, fd = self.b.request(self.me, {"op": "open", "name": "m"})
        self.assertFalse(reply["ok"])
        self.assertIsNone(fd)

    @unittest.skipUnless(hasattr(socket, "SO_PEERCRED"), "SO_PEERCRED is Linux only")
    def test_served_over_a_socket(self):
        path = os.path.join(self.dir, "s.sock")
        threading.Thread(target=self.b.serve, args=(path,), daemon=True).start()
        for _ in range(100):
            if os.path.exists(path):
                break
            threading.Event().wait(0.02)
        c = socket.socket(socket.AF_UNIX)
        c.connect(path)
        buf = bytearray()
        S.send_message(c, {"op": "open", "name": "m"})
        reply, fd = S.recv_message(c, buf)
        self.assertTrue(reply["ok"])
        self.assertIsNotNone(fd)
        S.send_message(c, {"op": "list"})
        self.assertEqual(S.recv_message(c, buf)[0]["segments"], {"m": True})
        os.close(fd)
        c.close()


if __name__ == "__main__":
    unittest.main(verbosity=1)
