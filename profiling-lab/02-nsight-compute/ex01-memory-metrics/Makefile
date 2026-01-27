# Makefile for memory metrics exercise
# Nsight tools paths (source scripts/setup-nsight-env.sh or use these directly)
NSIGHT_HOME ?= $(HOME)/envs/nsight_tools/nsight-compute-2025.4.1
NCU ?= $(NSIGHT_HOME)/ncu
NSYS ?= $(NSIGHT_HOME)/host/target-linux-x64/nsys

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
	$(NCU) --kernel-name coalesced_copy --section MemoryWorkloadAnalysis ./$(TARGET)
	@echo ""
	@echo "=== Profiling strided_copy ==="
	$(NCU) --kernel-name strided_copy --section MemoryWorkloadAnalysis ./$(TARGET)
	@echo ""
	@echo "=== Profiling random_copy ==="
	$(NCU) --kernel-name random_copy --section MemoryWorkloadAnalysis ./$(TARGET)

# Generate full report
report: $(TARGET)
	$(NCU) -o memory_metrics_report --set full ./$(TARGET)
	@echo "Report saved to memory_metrics_report.ncu-rep"
	@echo "Open with: ncu-ui memory_metrics_report.ncu-rep"

clean:
	rm -f $(TARGET) $(TARGET)_debug *.ncu-rep *.nsys-rep
