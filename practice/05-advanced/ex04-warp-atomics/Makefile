NVCC = nvcc
NVCC_FLAGS = -std=c++17

all: warp_atomics solution

warp_atomics: warp_atomics.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f warp_atomics solution

test: warp_atomics
	./warp_atomics

.PHONY: all clean test
