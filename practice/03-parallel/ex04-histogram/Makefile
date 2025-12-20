NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70

all: histogram

histogram: histogram.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f histogram solution

.PHONY: all clean solution
