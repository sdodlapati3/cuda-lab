NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70 -std=c++14

all: events

events: events.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f events solution

.PHONY: all clean solution
