# Undocumented driver tables

`cuGetExportTable(&table, &uuid)` returns tables of undocumented driver functions. cudart 12
calls it before any public API, and cuBLASLt calls a few more. The shim provides the tables
below.

| uuid prefix | name | used by | what the shim does |
|---|---|---|---|
| `6bd5fb6c` | CUDART_INTERFACE | cudart | loads modules from fatbins, returns the primary context, retains it with flags |
| `42d85a81` | TOOLS_TLS (3 slots on 570) | cudart, cuBLASLt | `[2](void **out, flag)` returns a tools object |
| `a094798c` | TOOLS_RUNTIME_CALLBACK_HOOKS | cudart | returns two buffers |
| `c693336e` | CONTEXT_LOCAL_STORAGE | cudart | stores, reads and deletes values keyed on the context |
| `263e8860` | CONTEXT_CHECKS | cudart | context check returns 0 |
| `d4082055` | INTEGRITY_CHECK | every copy of cudart, including static ones inside cuBLAS | computes the keyed hash; `tools/ic_check.py` compares it with the real driver |
| `0ca50b8c`, `195bcbf4`, `b10541e1` | context-create bypass, heap access, device extensions | none seen | provided as in ZLUDA |
| `21318c60` | 92 entries | cuBLASLt heuristics | `[4](obj, uint64_t *out, 0, 1)` sets `*out = 1`; other entries log and fail |
| `69a28e9f` | 26 entries | cudart, optional | not provided; cudart works without it |

The first seven come from ZLUDA's work (see NOTICE). We found the `TOOLS_TLS[2]` and
`21318c60[4]` behaviour with `tools/etspy.c`. It builds a proxy `libcuda.so.1` that loads the
real driver by absolute path, wraps each table entry in a logging function, and can dump the
memory that pointer arguments point to. Run it without CUPTI: CUPTI itself calls `21318c60`
thousands of times, which hides what cuBLASLt does.

These tables are not a stable interface. Each driver branch needs validation
([new-driver.md](new-driver.md)).
