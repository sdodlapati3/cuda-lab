NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70

all: scan

scan: scan.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f scan solution

.PHONY: all clean solution
