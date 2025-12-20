NVCC = nvcc
NVCC_FLAGS = -std=c++17 -rdc=true -lcudadevrt

all: cdp_patterns solution

cdp_patterns: cdp_patterns.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f cdp_patterns solution

test: cdp_patterns
	./cdp_patterns

.PHONY: all clean test
