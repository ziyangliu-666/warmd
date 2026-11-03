# Review of the nvproxy change for segments in gVisor

Scope: `scripts/gvisor_nvproxy_host_fd.py`, applied to gVisor's go branch of 2026-11-02. Version 1 ran on an RTX 3060 (Results 24, 25, 29). Version 2 adds a device check. It compiles for linux/amd64 and has not run on a GPU. Diff: `results/h31_patch_v2.diff`.

## What the change does

`ctrlHasFrontendFD` translates a sandbox fd into a host fd for RM control calls that carry an fd. Before the change it accepted only `frontendFD`, which nvproxy creates when the sandbox opens `/dev/nvidiactl`. After the change it also accepts a file that has a host fd. Then a control-device fd from outside the sandbox can carry an exported allocation in.

## Findings

1. **Type confusion in the driver, closed in version 2.** `nv_get_file_private` checks only that the inode's device number is 195:255. It does not check that the file is a character device or that it belongs to the NVIDIA driver, and then it uses the file's private data as its own structure. Version 1 let any host-backed file reach that check. A block device numbered 195:255 would put another driver's private data in its place. Version 2 rejects the fd unless fstat shows a character device 195:255. Exploitation would also need the host to give such a file to the sandbox.
2. **Files that version 1 accepted.** Three types have `HostFD()`: host files (`fsimpl/host`), their inodes, and nvproxy's `dmaBufFDWrapper`. A regular file was rejected by the driver in the negative test. Version 2 rejects all of them before the driver.
3. **The fd is the capability.** Any process that holds an exported fd can map the allocation. The broker must decide who receives it. The change does not widen this, but it lets the decision reach sandboxed tenants.
4. **`--host-uds=open` is broad.** The test let the sandbox connect to host sockets in its mounts. A deployment should donate the socket or fd to the sandbox directly instead.
5. **Object tracking is unchanged.** nvproxy did not record imported objects before the change either. An import through a host fd uses the same path afterwards.
6. **Sandbox save and restore.** A host fd is not saved. Restoring a sandbox that holds an imported segment is not supported and was not tested.

## Not covered

- No fuzzing and no review by gVisor maintainers.
- Version 2 has not run on hardware.
- Interaction with nvproxy's driver-version checks: version 1 ran with a forced version on driver 535, and on 570.195.03, which is supported.
