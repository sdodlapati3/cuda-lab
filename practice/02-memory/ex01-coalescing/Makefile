NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70

TARGET = coalescing
SOLUTION = solution

all: $(TARGET)

$(TARGET): coalescing.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(TARGET) $(SOLUTION)

.PHONY: all clean solution
