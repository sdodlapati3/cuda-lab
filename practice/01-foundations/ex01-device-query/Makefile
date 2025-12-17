# Makefile for CUDA exercises

NVCC = nvcc
NVCC_FLAGS = -O2

# Default target
all: device_query solution

device_query: device_query.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Run your implementation
run: device_query
	./device_query

# Run solution
run-solution: solution
	./solution

# Test against expected output
test:
	bash test.sh

# Clean build artifacts
clean:
	rm -f device_query solution *.o output.txt

.PHONY: all run run-solution test clean
