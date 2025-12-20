NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70

all: warp_reduction

warp_reduction: warp_reduction.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f warp_reduction solution

.PHONY: all clean solution
