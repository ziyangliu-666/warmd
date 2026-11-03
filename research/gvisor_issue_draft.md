<!-- Draft for google/gvisor, feature request template. Not posted. -->

# nvproxy: import a CUDA allocation exported from outside the sandbox

## Description

A sandboxed CUDA process cannot import device memory that a process outside the sandbox exported. The import fails with `CUDA_ERROR_INVALID_DEVICE` (101).

The use case is read-only model weights shared between tenants. A broker outside the sandbox creates the allocation with `cuMemCreate` and exports it with `cuMemExportToShareableHandle(..., CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)`. It sends the fd over a Unix socket. The client calls `cuMemImportFromShareableHandle`, then `cuMemMap` and `cuMemSetAccess` with `CU_MEM_ACCESS_FLAGS_PROT_READ`. Each client keeps its own context, so a fault stays in that client. No copy of the weights is made.

Outside gVisor this works across two runc containers, a second uid, and a client in an unprivileged user namespace. It also works when the exporter has exited before the import.

Inside gVisor, the import works when exporter and importer are in the same sandbox. It fails when the fd comes from outside.

Cause: the import is `NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD`, which carries an fd. `ctrlHasFrontendFD` looks up that fd in the sandbox's fd table and returns `EINVAL` unless the file is a `*frontendFD`. An fd received over a host Unix socket is a host file (`fsimpl/host`), so the call never reaches the driver.

Environment:

- runsc built from the `go` branch at cb99234 (2026-11-02), `--nvproxy=true --nvproxy-docker=true --host-uds=open`
- NVIDIA driver 570.195.03, RTX 3060, in a VM
- Docker with the NVIDIA container toolkit

Reproduction, about 80 lines of C: an owner creates and fills 1 GiB, exports it, and serves the fd on a Unix socket in a shared directory. A client receives the fd, imports, maps read-only, reads both ends, and tries a write. I can attach the program or put it in a gist.

```
docker run -d --gpus all -v /x:/x ubuntu:22.04 /x/xc owner /x/s 1024
docker run --rm --runtime=runsc --gpus all -v /x:/x ubuntu:22.04 /x/xc client /x/s 1024
```

| client | result |
|---|---|
| runc | import 0.07 ms, data correct, write rejected |
| runsc, owner in the same sandbox | import 0.44 ms, data correct, write rejected |
| runsc, owner outside | `cuMemImportFromShareableHandle` returns 101 |

## Is this feature related to a specific bug?

No. I did not find an existing issue.

## Do you have a specific solution in mind?

Let `ctrlHasFrontendFD` accept a host file when it is the NVIDIA control device:

```go
var hostFD int32
switch f := ctlFileGeneric.Impl().(type) {
case *frontendFD:
	hostFD = f.hostFD
case interface{ HostFD() int }:
	// Received from outside the sandbox. The driver checks only the device
	// number, then uses the file's private data, so check the type here.
	var st unix.Stat_t
	if err := unix.Fstat(f.HostFD(), &st); err != nil {
		return 0, linuxerr.EINVAL
	}
	if st.Mode&unix.S_IFMT != unix.S_IFCHR ||
		unix.Major(uint64(st.Rdev)) != nvgpu.NV_MAJOR_DEVICE_NUMBER ||
		unix.Minor(uint64(st.Rdev)) != nvgpu.NV_MINOR_DEVICE_NUMBER_CONTROL_DEVICE {
		return 0, linuxerr.EINVAL
	}
	hostFD = int32(f.HostFD())
default:
	return 0, linuxerr.EINVAL
}
```

With this change, the outside-owner case passes: import 0.60 ms, data correct, write rejected.

An earlier version without the fstat check passed two more tests. A PyTorch model on the imported weights gave the same output as in runc. A kernel store into the read-only weights faulted only the writer, in all four combinations of runc and runsc for writer and bystander. I have not repeated these two with the fstat check.

Why the fstat check: `nv_get_file_private` in the open kernel modules checks only that the inode's device number is 195:255. It does not check that the file is a character device that belongs to the driver, and then it uses `filp->private_data` as its own structure. Without the check, the change would let any host-backed file in the sandbox reach that path.

Negative tests: with and without the check, host fds of `/dev/null`, `/dev/nvidia0`, a fresh `/dev/nvidiactl`, and a regular file are all rejected. The case that the check adds protection for, a block device numbered 195:255, cannot be opened without a driver behind it, so I could not test it.

Things I am not sure about:

1. Is `--host-uds=open` the right way to receive the fd? A donated fd, through the OCI spec or a runsc flag, may fit gVisor's model better.
2. Should this be behind a flag, for example `--nvproxy-allow-host-ctl-fds`?
3. A host fd is not saved, so a sandbox that holds an imported allocation cannot be checkpointed. Should nvproxy reject `runsc checkpoint` in that state?
4. The other users of `ctrlHasFrontendFD` (export, get info) also change. I tested only import.

Alternatives considered:

- Run the exporter inside the sandbox. This works, but it makes sharing between sandboxes impossible.
- Copy the weights into each sandbox. This works, but each tenant then pays the full device memory.

I can send this as a PR, with a test if you can point me to the right place for nvproxy tests.
