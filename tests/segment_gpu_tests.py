"""GPU tests for warmd.segments. Needs a CUDA GPU, PyTorch and `make`.

    python3 tests/segment_gpu_tests.py

Each case runs a broker in a subprocess and clients in separate processes.
"""
import json
import multiprocessing as mp
import os
import struct
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "python"))


def make_file(path):
    import torch
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(0)
    save_file({"w": torch.randn(512, 512, generator=g), "b": torch.randn(512, generator=g).half()}, path)


def client_sum(sock, q):
    from warmd.segments import Client
    c = Client(sock)
    t = c.open("m")
    a = float((t["w"] @ t["w"]).sum())
    c.park("m")
    c.resume("m")
    q.put((a, float((t["w"] @ t["w"]).sum())))


def client_write(sock, q):
    import torch
    from warmd.segments import Client
    t = Client(sock).open("m")
    try:
        t["w"].fill_(9.0)
        torch.cuda.synchronize()
        q.put("store completed")
    except RuntimeError as e:
        q.put("faulted" if "illegal memory access" in str(e) else str(e)[:80])


class SegmentGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.file = os.path.join(cls.dir, "m.safetensors")
        make_file(cls.file)
        cls.sock = os.path.join(cls.dir, "seg.sock")
        env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "python"))
        cls.broker = subprocess.Popen([sys.executable, "-m", "warmd.segments", "--socket", cls.sock, "--file", f"m={cls.file}"], env=env)
        for _ in range(200):
            if os.path.exists(cls.sock):
                break
            time.sleep(0.05)
        cls.ctx = mp.get_context("spawn")

    @classmethod
    def tearDownClass(cls):
        cls.broker.kill()

    def run_child(self, fn):
        q = self.ctx.Queue()
        p = self.ctx.Process(target=fn, args=(self.sock, q))
        p.start()
        r = q.get(timeout=120)
        p.join()
        return r

    def test_two_clients_see_the_same_data_after_park_and_resume(self):
        a1, b1 = self.run_child(client_sum)
        a2, _ = self.run_child(client_sum)
        self.assertEqual((a1, a1), (b1, a2))

    def test_a_store_faults_the_writer_and_not_the_next_client(self):
        self.assertEqual(self.run_child(client_write), "faulted")
        a, b = self.run_child(client_sum)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=1)
