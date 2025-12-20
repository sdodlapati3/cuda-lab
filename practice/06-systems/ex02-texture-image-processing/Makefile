NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80

all: texture_processor solution

texture_processor: texture_processor.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f texture_processor solution

test: solution
	./solution

.PHONY: all clean test
