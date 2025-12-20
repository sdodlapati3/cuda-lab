NVCC = nvcc
NVCC_FLAGS = -std=c++17 -rdc=true -lcudadevrt

all: cdp_octree solution

cdp_octree: cdp_octree.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f cdp_octree solution

test: cdp_octree
	./cdp_octree

.PHONY: all clean test
