# Makefile for compute metrics exercise
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
	ncu --section ComputeWorkloadAnalysis ./$(TARGET)

# Profile occupancy
occupancy: $(TARGET)
	@echo "=== Profiling Occupancy ==="
	ncu --section Occupancy ./$(TARGET)

# Profile warp efficiency
warp: $(TARGET)
	@echo "=== Profiling Warp State ==="
	ncu --kernel-name divergent_kernel --section WarpStateStatistics ./$(TARGET)
	ncu --kernel-name uniform_kernel --section WarpStateStatistics ./$(TARGET)

# Full report
report: $(TARGET)
	ncu -o compute_metrics_report --set full ./$(TARGET)
	@echo "Report saved to compute_metrics_report.ncu-rep"

clean:
	rm -f $(TARGET) $(TARGET)_debug *.ncu-rep *.nsys-rep
