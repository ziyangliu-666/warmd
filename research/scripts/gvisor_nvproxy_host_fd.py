#!/usr/bin/env python3
"""Patch gVisor nvproxy to accept a control-device fd received from outside the sandbox.

ctrlHasFrontendFD returned EINVAL for any fd that was not opened through nvproxy in this sandbox, so
a device-memory segment could not enter a sandbox. With this change, nvproxy passes a host fd to
the driver if fstat shows the NVIDIA control character device. The driver's own check
(nv_get_file_private) tests only the device number and then trusts the file's private data, so a
block device with the same number would confuse types in the kernel. The fstat check closes that.

Usage: gvisor_nvproxy_host_fd.py <path to pkg/sentry/devices/nvproxy/frontend.go>
"""
import sys
path = sys.argv[1]; s = open(path).read()
old = """	ctlFile, ok := ctlFileGeneric.Impl().(*frontendFD)
	if !ok {
		return 0, linuxerr.EINVAL
	}

	ctrlParams.SetFrontendFD(ctlFile.hostFD)
	n, err := rmControlInvoke(fi, ioctlParams, ctrlParams)
	ctrlParams.SetFrontendFD(origFD)"""
new = """	var hostFD int32
	switch f := ctlFileGeneric.Impl().(type) {
	case *frontendFD:
		hostFD = f.hostFD
	case interface{ HostFD() int }:
		// Received from outside the sandbox. The driver checks only the device number and then
		// uses the file's private data, so accept only the NVIDIA control character device here.
		var st unix.Stat_t
		if err := unix.Fstat(f.HostFD(), &st); err != nil {
			return 0, linuxerr.EINVAL
		}
		if st.Mode&unix.S_IFMT != unix.S_IFCHR || unix.Major(uint64(st.Rdev)) != nvgpu.NV_MAJOR_DEVICE_NUMBER || unix.Minor(uint64(st.Rdev)) != nvgpu.NV_MINOR_DEVICE_NUMBER_CONTROL_DEVICE {
			return 0, linuxerr.EINVAL
		}
		hostFD = int32(f.HostFD())
	default:
		return 0, linuxerr.EINVAL
	}

	ctrlParams.SetFrontendFD(hostFD)
	n, err := rmControlInvoke(fi, ioctlParams, ctrlParams)
	ctrlParams.SetFrontendFD(origFD)"""
if old not in s:
    sys.exit("expected code not found")
open(path, "w").write(s.replace(old, new, 1)); print("patched")
