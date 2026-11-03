"""Send a host fd of a chosen file over a Unix socket, the path that the nvproxy change opens."""
import os, socket, sys
path, target = sys.argv[1], sys.argv[2]
if os.path.exists(path):
    os.unlink(path)
s = socket.socket(socket.AF_UNIX); s.bind(path); os.chmod(path, 0o666); s.listen(4)
fd = os.open(target, os.O_RDONLY)
while True:
    c, _ = s.accept(); socket.send_fds(c, [b"f"], [fd]); c.close()
