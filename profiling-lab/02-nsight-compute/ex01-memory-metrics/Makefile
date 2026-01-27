# Makefile for memory metrics exercise
NVCC = nvcc
NVCC_FLAGS = -O3 -lineinfo
DEBUG_FLAGS = -g -G

TARGET = memory_bandwidth
SRC = memory_bandwidth.cu

.PHONY: all clean debug profile

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

debug: $(SRC)
	$(NVCC) $(DEBUG_FLAGS) -o $(TARGET)_debug $<

# Profile with Nsight Compute
profile: $(TARGET)
	@echo "=== Profiling coalesced_copy ==="
	ncu --kernel-name coalesced_copy --section MemoryWorkloadAnalysis ./$(TARGET)
	@echo ""
	@echo "=== Profiling strided_copy ==="
	ncu --kernel-name strided_copy --section MemoryWorkloadAnalysis ./$(TARGET)
	@echo ""
	@echo "=== Profiling random_copy ==="
	ncu --kernel-name random_copy --section MemoryWorkloadAnalysis ./$(TARGET)

# Generate full report
report: $(TARGET)
	ncu -o memory_metrics_report --set full ./$(TARGET)
	@echo "Report saved to memory_metrics_report.ncu-rep"
	@echo "Open with: ncu-ui memory_metrics_report.ncu-rep"

clean:
	rm -f $(TARGET) $(TARGET)_debug *.ncu-rep *.nsys-rep
