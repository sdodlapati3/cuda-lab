NVCC = nvcc
NVCC_FLAGS = -O3 -lineinfo -std=c++17 --use_fast_math

GPU_ARCH ?= sm_80

all: softmax

softmax: softmax_online.cu
	$(NVCC) $(NVCC_FLAGS) -arch=$(GPU_ARCH) -o $@ $<

run: softmax
	./softmax

# Profile memory traffic
profile: softmax
	ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,sm__throughput.avg_pct_of_peak_sustained_elapsed ./softmax

clean:
	rm -f softmax *.o *.ncu-rep

.PHONY: all run profile clean
