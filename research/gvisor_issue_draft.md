<!-- Draft for google/gvisor, feature request. Not posted. -->

# nvproxy: allow importing CUDA memory exported from outside the sandbox

## Description

In a sandbox, `cuMemImportFromShareableHandle` fails with `CUDA_ERROR_INVALID_DEVICE` when the fd comes from a process outside the sandbox. The same import works in runc, and it works in gVisor when both processes are in one sandbox.

The import is the RM control `NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD`. `ctrlHasFrontendFD` returns `EINVAL` unless the fd is a `*frontendFD`. An fd received over a host Unix socket is a host file, so the call never reaches the driver.

Use case: one process exports read-only model weights (`cuMemExportToShareableHandle`, POSIX fd) and several sandboxes map them without a copy.

To reproduce, with the exporter in runc and the importer in runsc (`--nvproxy=true --host-uds=open`), send the fd over a Unix socket in a shared directory. I can attach an 80-line C program. Tested on the `go` branch at cb99234 with driver 570.195.03.

## Is this feature related to a specific bug?

No.

## Do you have a specific solution in mind?

Accept a host file in `ctrlHasFrontendFD` if it is the NVIDIA control device:

```go
case interface{ HostFD() int }:
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
```

With this change the import works and the data is correct. Host fds of other files are still rejected.

The fstat check matters: the driver's `nv_get_file_private` checks only the device number, then uses the file's private data as its own.

Question: is a donated fd better than `--host-uds=open` for this? I can send a PR either way.
