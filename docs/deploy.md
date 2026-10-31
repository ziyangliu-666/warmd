# Deploying warmd

Run one daemon per GPU. A task process needs only the daemon's socket and the shim library. It
does not need the GPU device nodes, the NVIDIA container runtime, or a GPU resource request.

## systemd

```console
$ sudo make install
$ sudo mkdir -p /etc/warmd
$ sudo cp deploy/gpu0.conf.example /etc/warmd/gpu0.conf      # set socket = /run/warmd/gpu0.sock
$ sudo cp deploy/systemd/warmd@.service /etc/systemd/system/
$ sudo systemctl enable --now warmd@0
```

`make install` puts the daemon in `/usr/local/bin/warmd` and the shim in
`/usr/local/lib/warmd/libcuda.so.1`. For GPU 1 add `/etc/warmd/gpu1.conf` and enable
`warmd@1`.

Clients set two variables:

```console
$ export WARMD_SOCKET=/run/warmd/gpu0.sock
$ export LD_LIBRARY_PATH=/usr/local/lib/warmd
```

The socket file is mode 0666 by default, and the daemon checks the uid of every connection
with `SO_PEERCRED`. The daemon's own uid and root are allowed. Add more with `allow-uid` lines
or allow everyone with `allow-all = true`. Keep `nvidia-persistenced` running so the driver
stays loaded between restarts.

## Containers

Build the two images defined in `deploy/docker/Dockerfile`:

```console
$ docker build -f deploy/docker/Dockerfile --target warmd -t warmd:0.1.0 .
$ docker build -f deploy/docker/Dockerfile --target warmd-client-libs -t warmd-client-libs:0.1.0 .
```

The daemon container gets the GPU and publishes its socket directory on the host:

```console
$ docker run -d --gpus '"device=0"' -v /run/warmd:/run/warmd warmd:0.1.0
```

Task containers mount the socket directory and the shim. They do not use `--gpus`:

```console
$ docker run --rm -v /run/warmd:/run/warmd -v /usr/local/lib/warmd:/opt/warmd/lib:ro \
    -e WARMD_SOCKET=/run/warmd/gpu0.sock -e LD_LIBRARY_PATH=/opt/warmd/lib \
    pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
    python -c "import torch; print(torch.ones(2, device='cuda'))"
```

This works across containers because session memory is a `memfd` passed over the unix socket.
There is no shared `/dev/shm` and no shared IPC namespace. The uid check sees the uid of the
process in the container (often 0, which is allowed). With user namespaces, use `allow-uid` or
`allow-all`.

`deploy/cdi/warmd.yaml` is a CDI spec. With a CDI-aware runtime,
`podman run --device warmd.dev/gpu=0 ...` adds the mounts and variables.

## Kubernetes

`deploy/k8s/warmd-daemonset.yaml` runs the daemon on GPU nodes (it requests one
`nvidia.com/gpu`) and puts the socket and the shim on the node under `/run/warmd` and
`/usr/local/lib/warmd`. `deploy/k8s/task-pod-example.yaml` shows a task pod: two hostPath
mounts and two environment variables, no GPU request.

These manifests and the CDI spec have not been tested on a cluster yet.

Task pods request no GPU, so the scheduler does not count them. Limit how many run per node in
your own queue for now. A device plugin that advertises one resource per session is planned.

## Sizing

| setting | effect |
|---|---|
| `sessions` | maximum concurrent task processes. Each costs `rpc-mb + ring-mb` (16 MB by default) of host memory, pinned once when a worker starts: 64 sessions pin 1 GB. |
| `standbys` | each standby worker holds its own CUDA context (0.4 to 0.9 GB of GPU memory on an A100). One is enough for fast failover. With 0, recovery after a fault takes 1 to 2 s. |
| `threads` | dispatch threads in the active worker. 4 is enough for most loads. |
| `quota-mb` | device memory limit per session. PyTorch reports it as the device's total memory. |
