NVCC = nvcc
NVCC_FLAGS = -std=c++17

all: cg_reduction solution

cg_reduction: cg_reduction.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f cg_reduction solution

test: cg_reduction
	./cg_reduction

.PHONY: all clean test
