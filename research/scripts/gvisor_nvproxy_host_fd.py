#!/usr/bin/env python3
"""Patch gVisor nvproxy to accept a control-device fd received from outside the sandbox.

ctrlHasFrontendFD returned EINVAL for any fd that was not opened through nvproxy in this sandbox, so
a device-memory segment could not enter a sandbox. With this change, nvproxy passes the host fd to
the driver. The driver still checks that the fd is the control device (nv_get_file_private), so a
host fd of any other kind fails there.

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
		// Received from outside the sandbox. The driver checks that it is the control device.
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
