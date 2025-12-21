NVCC = nvcc
NVCC_FLAGS = -O3 -lineinfo -std=c++17

GPU_ARCH ?= sm_80

all: transpose

transpose: transpose_tiled.cu
	$(NVCC) $(NVCC_FLAGS) -arch=$(GPU_ARCH) -o $@ $<

run: transpose
	./transpose

# Profile for bank conflicts
profile: transpose
	ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./transpose

clean:
	rm -f transpose *.o *.ncu-rep

.PHONY: all run profile clean
