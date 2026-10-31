# Deploying

Run one warmd per GPU. Clients need the socket and the client library, nothing else: no GPU
device nodes, no NVIDIA container runtime, no GPU resource request.

## systemd

```console
# make install
# mkdir -p /etc/warmd && cp contrib/gpu0.conf.example /etc/warmd/gpu0.conf
# cp contrib/systemd/warmd@.service /etc/systemd/system/
# systemctl enable --now warmd@0
```

`make install` puts the daemon in `/usr/local/bin` and the client library in
`/usr/local/lib/warmd`. Set `socket = /run/warmd/gpu0.sock` in the config. Clients then use

```console
$ export WARMD_SOCKET=/run/warmd/gpu0.sock LD_LIBRARY_PATH=/usr/local/lib/warmd
```

Only root and the daemon's uid may connect unless the config adds `allow-uid` lines or
`allow-all = true`. Run `nvidia-persistenced` so the driver stays loaded.

## Containers

```console
$ docker build -f contrib/docker/Dockerfile --target warmd -t warmd .
$ docker build -f contrib/docker/Dockerfile --target warmd-client-libs -t warmd-client-libs .
$ docker run -d --gpus device=0 -v /run/warmd:/run/warmd warmd
```

A client container mounts the socket directory and the library and needs no `--gpus`:

```console
$ docker run --rm -v /run/warmd:/run/warmd -v /usr/local/lib/warmd:/opt/warmd/lib:ro \
    -e WARMD_SOCKET=/run/warmd/gpu0.sock -e LD_LIBRARY_PATH=/opt/warmd/lib \
    pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime python -c 'import torch; print(torch.ones(2, device="cuda"))'
```

Session memory is a memfd sent over the socket, so containers need no shared `/dev/shm` or IPC
namespace. With user namespaces, the daemon sees the container's mapped uid; allow it.

`contrib/cdi/warmd.yaml` is a CDI spec for `--device warmd.dev/gpu=0`.

## Kubernetes

`contrib/k8s/warmd-daemonset.yaml` runs the daemon on GPU nodes and publishes `/run/warmd` and
the client library on each node. `contrib/k8s/task-pod-example.yaml` is a client pod. These files
and the CDI spec have not been tested on a cluster.

Client pods request no GPU, so the scheduler does not limit how many land on a node. Limit them
yourself until there is a device plugin for sessions.
