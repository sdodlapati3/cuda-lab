NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70

all: occupancy

occupancy: occupancy.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f occupancy solution

.PHONY: all clean solution
