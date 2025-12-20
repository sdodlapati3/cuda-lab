NVCC = nvcc
NVCC_FLAGS = -std=c++17

all: cg_scan solution

cg_scan: cg_scan.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f cg_scan solution

test: cg_scan
	./cg_scan

.PHONY: all clean test
