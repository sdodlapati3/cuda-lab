NVCC = nvcc
NVCC_FLAGS = -std=c++17

all: cuda_graphs solution

cuda_graphs: cuda_graphs.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f cuda_graphs solution

test: cuda_graphs
	./cuda_graphs

.PHONY: all clean test
