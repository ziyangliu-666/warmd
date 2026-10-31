CUDA    ?= /usr/local/cuda
CC      ?= gcc
CFLAGS  ?= -O2 -g -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-missing-field-initializers -Wno-format-truncation -Wno-type-limits
PREFIX  ?= /usr/local
LDFLAGS_EXTRA ?=
B       := build

.PHONY: all clean install test bench tools
all: $(B)/warmd $(B)/lib/libcuda.so.1 $(B)/demo $(B)/driver_features $(B)/driver_features_native

$(B)/warmd: host/warmd.c host/worker.c host/common.h proto/warmd_proto.h
	@mkdir -p $(B)
	$(CC) $(CFLAGS) -I$(CUDA)/include host/warmd.c host/worker.c -o $@ $(LDFLAGS_EXTRA) -lcuda -lpthread

# the shim: a drop-in libcuda.so.1 (put $(B)/lib on LD_LIBRARY_PATH of task processes)
$(B)/lib/libcuda.so.1: shim/shim.c shim/dark.c shim/stubs.c shim/shim.map proto/warmd_proto.h
	@mkdir -p $(B)/lib
	$(CC) $(CFLAGS) -fPIC -shared shim/shim.c shim/stubs.c -o $@ -Wl,--version-script=shim/shim.map -Wl,-soname,libcuda.so.1 -lpthread -ldl
	ln -sf libcuda.so.1 $(B)/lib/libcuda.so

# a driver-API program linked against the shim (used by tests)
$(B)/demo: tests/demo.c $(B)/lib/libcuda.so.1
	$(CC) $(CFLAGS) tests/demo.c -o $@ -I$(CUDA)/include -L$(B)/lib -lcuda -Wl,-rpath,'$$ORIGIN/lib' -lpthread

$(B)/driver_features: tests/driver_features.c $(B)/lib/libcuda.so.1
	$(CC) $(CFLAGS) $< -o $@ -I$(CUDA)/include -L$(B)/lib -lcuda -Wl,-rpath,'$$ORIGIN/lib'
$(B)/driver_features_native: tests/driver_features.c
	$(CC) $(CFLAGS) $< -o $@ -I$(CUDA)/include $(LDFLAGS_EXTRA) -lcuda

tools: $(B)/libcupti_inject.so $(B)/librmtrace.so
CUPTI ?= $(shell python3 -c "import nvidia.cuda_cupti,os;print(os.path.dirname(nvidia.cuda_cupti.__file__))" 2>/dev/null)
$(B)/libcupti_inject.so: tools/cupti_inject.c
	$(CC) -O2 -shared -fPIC $< -o $@ -I$(CUDA)/include -I$(CUPTI)/include $(CUPTI)/lib/libcupti.so.12 -Wl,-rpath,$(CUPTI)/lib -lpthread
$(B)/librmtrace.so: tools/rmtrace.c
	$(CC) -O2 -shared -fPIC $< -o $@ -ldl

install: all
	install -D -m 755 $(B)/warmd $(DESTDIR)$(PREFIX)/bin/warmd
	install -D -m 755 $(B)/lib/libcuda.so.1 $(DESTDIR)$(PREFIX)/lib/warmd/libcuda.so.1
	ln -sf libcuda.so.1 $(DESTDIR)$(PREFIX)/lib/warmd/libcuda.so

test: all
	tests/run.sh
bench: all
	bench/run.sh
clean:
	rm -rf $(B)
