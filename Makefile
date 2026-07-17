# overhang — run massive models on your small Mac
# Engines build with clang; OpenMP via Homebrew libomp if present; Metal 4
# GPU prefill for olmoe via `make olmoe-metal`.
CC      = clang
OMPDIR := $(shell brew --prefix libomp 2>/dev/null)
ifneq ($(wildcard $(OMPDIR)/include/omp.h),)
OMPC    = -Xclang -fopenmp -I$(OMPDIR)/include
OMPL    = -L$(OMPDIR)/lib -lomp
endif
CFLAGS  = -O3 $(OMPC) -Iinclude -Wall -Wno-unused-function
LDFLAGS = -lm $(OMPL)

all: qwen olmoe gemma

qwen: engines/qwen.c include/st.h include/json.h
	$(CC) $(CFLAGS) engines/qwen.c -o qwen $(LDFLAGS)

olmoe: engines/olmoe.c include/st.h include/json.h
	$(CC) $(CFLAGS) engines/olmoe.c -o olmoe $(LDFLAGS)

gemma: engines/gemma.c include/st.h include/json.h
	$(CC) $(CFLAGS) engines/gemma.c -o gemma $(LDFLAGS)

olmoe-metal: engines/olmoe.c metal/om_metal.mm metal/om_metal.h
	$(CC) $(CFLAGS) -Imetal -DOM_METAL -c engines/olmoe.c -o olmoe_c.o
	clang++ -O3 -ObjC++ -c metal/om_metal.mm -o om_metal.o
	clang++ olmoe_c.o om_metal.o -o olmoe-metal $(LDFLAGS) -framework Metal -framework Foundation

gemma-metal: engines/gemma.c metal/om_metal.mm metal/om_metal.h
	$(CC) $(CFLAGS) -Imetal -DOM_METAL -c engines/gemma.c -o gemma_c.o
	clang++ -O3 -ObjC++ -c metal/om_metal.mm -o om_metal.o
	clang++ gemma_c.o om_metal.o -o gemma-metal $(LDFLAGS) -framework Metal -framework Foundation

check-oracle: qwen
	SNAP=oracle/qwen_tiny ./qwen oracle/ref_qwen.json

check-oracle-gemma: gemma
	SNAP=oracle/gemma_tiny ./gemma oracle/ref_gemma.json

clean:
	rm -f qwen olmoe gemma olmoe-metal gemma-metal *.o
.PHONY: all check-oracle check-oracle-gemma clean
