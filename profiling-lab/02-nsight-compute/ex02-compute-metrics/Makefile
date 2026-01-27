# Makefile for compute metrics exercise
# Nsight tools paths (source scripts/setup-nsight-env.sh or use these directly)
NSIGHT_HOME ?= $(HOME)/envs/nsight_tools/nsight-compute-2025.4.1
NCU ?= $(NSIGHT_HOME)/ncu
NSYS ?= $(NSIGHT_HOME)/host/target-linux-x64/nsys

NVCC = nvcc
NVCC_FLAGS = -O3 -lineinfo --use_fast_math
DEBUG_FLAGS = -g -G

TARGET = compute_metrics
SRC = compute_metrics.cu

.PHONY: all clean debug profile

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

debug: $(SRC)
	$(NVCC) $(DEBUG_FLAGS) -o $(TARGET)_debug $<

# Profile compute metrics
profile: $(TARGET)
	@echo "=== Profiling Compute Workload ==="
	$(NCU) --section ComputeWorkloadAnalysis ./$(TARGET)

# Profile occupancy
occupancy: $(TARGET)
	@echo "=== Profiling Occupancy ==="
	$(NCU) --section Occupancy ./$(TARGET)

# Profile warp efficiency
warp: $(TARGET)
	@echo "=== Profiling Warp State ==="
	$(NCU) --kernel-name divergent_kernel --section WarpStateStatistics ./$(TARGET)
	$(NCU) --kernel-name uniform_kernel --section WarpStateStatistics ./$(TARGET)

# Full report
report: $(TARGET)
	$(NCU) -o compute_metrics_report --set full ./$(TARGET)
	@echo "Report saved to compute_metrics_report.ncu-rep"

clean:
	rm -f $(TARGET) $(TARGET)_debug *.ncu-rep *.nsys-rep
