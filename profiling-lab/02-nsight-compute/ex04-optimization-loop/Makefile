# Makefile for optimization loop exercise
NVCC = nvcc
NVCC_FLAGS = -O3 -lineinfo
DEBUG_FLAGS = -g -G

TARGET = optimization_loop
SRC = optimization_loop.cu

.PHONY: all clean debug profile compare

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

debug: $(SRC)
	$(NVCC) $(DEBUG_FLAGS) -o $(TARGET)_debug $<

# Profile all versions and save reports
profile-all: $(TARGET)
	@echo "=== Profiling all transpose versions ==="
	ncu --kernel-name transpose_v0_naive -o v0_naive --set full ./$(TARGET)
	ncu --kernel-name transpose_v1_shared -o v1_shared --set full ./$(TARGET)
	ncu --kernel-name transpose_v2_padded -o v2_padded --set full ./$(TARGET)
	ncu --kernel-name transpose_v3_coarsened -o v3_coarsened --set full ./$(TARGET)
	ncu --kernel-name transpose_v4_final -o v4_final --set full ./$(TARGET)
	@echo ""
	@echo "Reports saved. Compare with ncu-ui or:"
	@echo "ncu --import v0_naive.ncu-rep --page details"

# Quick comparison of baseline vs optimized
compare: $(TARGET)
	@echo "=== Comparing V0 (naive) vs V4 (optimized) ==="
	ncu --kernel-name transpose_v0_naive --section MemoryWorkloadAnalysis ./$(TARGET)
	@echo "---"
	ncu --kernel-name transpose_v4_final --section MemoryWorkloadAnalysis ./$(TARGET)

# Check for bank conflicts
bank-conflicts: $(TARGET)
	@echo "=== Checking shared memory bank conflicts ==="
	@echo "V1 (with bank conflicts):"
	ncu --kernel-name transpose_v1_shared --section MemoryWorkloadAnalysis_Tables ./$(TARGET)
	@echo ""
	@echo "V2 (padded - no bank conflicts):"
	ncu --kernel-name transpose_v2_padded --section MemoryWorkloadAnalysis_Tables ./$(TARGET)

# Memory efficiency analysis
memory-efficiency: $(TARGET)
	ncu --kernel-name transpose_v0_naive --metrics \
		l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
		l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
		smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
		smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct \
		./$(TARGET)

clean:
	rm -f $(TARGET) $(TARGET)_debug *.ncu-rep *.nsys-rep v*.ncu-rep
