# Two images from one Dockerfile:
#   warmd   - the daemon (needs the GPU: run with --gpus / the NVIDIA runtime)
#   warmd-client-libs - just /opt/warmd/lib/libcuda.so.1 to copy or mount into task images
# Task containers need NO GPU device nodes: they only need the socket and the shim library.
ARG CUDA=12.4.1
FROM nvidia/cuda:${CUDA}-devel-ubuntu22.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends make gcc && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN make build/warmd build/lib/libcuda.so.1 CFLAGS="-O2 -g -Wall" LDFLAGS_EXTRA="-L/usr/local/cuda/lib64/stubs"

FROM nvidia/cuda:${CUDA}-base-ubuntu22.04 AS warmd
COPY --from=build /src/build/warmd /usr/local/bin/warmd
COPY deploy/gpu0.conf.example /etc/warmd/gpu0.conf
VOLUME /run/warmd
ENTRYPOINT ["/usr/local/bin/warmd", "--config", "/etc/warmd/gpu0.conf", "--allow-all"]

FROM scratch AS warmd-client-libs
COPY --from=build /src/build/lib/libcuda.so.1 /opt/warmd/lib/libcuda.so.1
