NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70

TARGET = shared_memory
SOLUTION = solution

all: $(TARGET)

$(TARGET): shared_memory.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(TARGET) $(SOLUTION)

.PHONY: all clean solution
