# Makefile for roofline practice exercise
# Nsight tools paths (source scripts/setup-nsight-env.sh or use these directly)
NSIGHT_HOME ?= $(HOME)/envs/nsight_tools/nsight-compute-2025.4.1
NCU ?= $(NSIGHT_HOME)/ncu
NSYS ?= $(NSIGHT_HOME)/host/target-linux-x64/nsys

NVCC = nvcc
NVCC_FLAGS = -O3 -lineinfo --use_fast_math
DEBUG_FLAGS = -g -G

TARGET = roofline_kernels
SRC = roofline_kernels.cu

.PHONY: all clean debug roofline

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

debug: $(SRC)
	$(NVCC) $(DEBUG_FLAGS) -o $(TARGET)_debug $<

# Generate roofline chart
roofline: $(TARGET)
	$(NCU) --set roofline -o roofline_report ./$(TARGET)
	@echo ""
	@echo "Report saved to roofline_report.ncu-rep"
	@echo "Open with: ncu-ui roofline_report.ncu-rep"

# Profile individual kernels for roofline
profile-memory-bound: $(TARGET)
	@echo "=== Memory-Bound Kernels ==="
	$(NCU) --kernel-name stream_copy --set roofline ./$(TARGET)
	$(NCU) --kernel-name stream_triad --set roofline ./$(TARGET)

profile-compute-bound: $(TARGET)
	@echo "=== Compute-Bound Kernels ==="
	$(NCU) --kernel-name high_compute --set roofline ./$(TARGET)
	$(NCU) --kernel-name very_high_compute --set roofline ./$(TARGET)

# Full analysis
full-report: $(TARGET)
	$(NCU) -o roofline_full --set full ./$(TARGET)
	@echo "Full report: roofline_full.ncu-rep"

clean:
	rm -f $(TARGET) $(TARGET)_debug *.ncu-rep *.nsys-rep
