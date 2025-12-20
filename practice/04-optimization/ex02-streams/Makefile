NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70

all: streams

streams: streams.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f streams solution

.PHONY: all clean solution
