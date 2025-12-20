NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70

TARGET = bank_conflicts
SOLUTION = solution

all: $(TARGET)

$(TARGET): bank_conflicts.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(TARGET) $(SOLUTION)

.PHONY: all clean solution
