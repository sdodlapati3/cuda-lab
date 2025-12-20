NVCC = nvcc
NVCC_FLAGS = -std=c++17 -rdc=true -lcudadevrt

all: cdp_quicksort solution

cdp_quicksort: cdp_quicksort.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f cdp_quicksort solution

test: cdp_quicksort
	./cdp_quicksort

.PHONY: all clean test
