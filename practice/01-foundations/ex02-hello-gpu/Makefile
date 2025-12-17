# Makefile for CUDA exercises

NVCC = nvcc
NVCC_FLAGS = -O2

# Default target
all: hello_gpu solution

hello_gpu: hello_gpu.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Run your implementation
run: hello_gpu
	./hello_gpu

# Run solution
run-solution: solution
	./solution

# Test against expected output
test:
	bash test.sh

# Clean build artifacts
clean:
	rm -f hello_gpu solution *.o output.txt

.PHONY: all run run-solution test clean
