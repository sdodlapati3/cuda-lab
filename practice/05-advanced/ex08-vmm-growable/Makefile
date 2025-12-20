NVCC = nvcc
NVCC_FLAGS = -std=c++17

all: vmm_growable solution

vmm_growable: vmm_growable.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ -lcuda

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ -lcuda

clean:
	rm -f vmm_growable solution

test: vmm_growable
	./vmm_growable

.PHONY: all clean test
