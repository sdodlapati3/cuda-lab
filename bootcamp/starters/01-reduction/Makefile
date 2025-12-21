NVCC = nvcc
NVCC_FLAGS = -O3 -lineinfo -arch=sm_80 -std=c++17

# Detect GPU architecture (fallback to sm_80)
GPU_ARCH ?= sm_80

all: reduction

reduction: reduction_warp.cu
	$(NVCC) $(NVCC_FLAGS) -arch=$(GPU_ARCH) -o $@ $<

run: reduction
	./reduction

profile: reduction
	ncu --set full -o reduction_profile ./reduction

clean:
	rm -f reduction *.o *.qdrep *.ncu-rep

.PHONY: all run profile clean
