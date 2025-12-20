#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Simple kernel to fill buffer
__global__ void fillBuffer(int* data, int n, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

// Simple kernel to sum buffer
__global__ void sumBuffer(int* data, int n, long long* result) {
    __shared__ long long sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? data[idx] : 0;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(result, sdata[0]);
}

// TODO: Implement VMM-based growable buffer class
// Requirements:
// 1. Reserve large virtual address range (e.g., 1GB)
// 2. Initially map small physical memory (e.g., 1MB)
// 3. Implement grow() to map additional physical memory
// 4. No copying required when growing!

class GrowableBuffer {
    CUdeviceptr ptr;
    size_t reservedSize;   // Virtual address reservation
    size_t mappedSize;     // Currently mapped physical memory
    size_t granularity;    // Allocation granularity
    
public:
    GrowableBuffer(size_t maxSize) {
        // TODO: Initialize CUDA driver API
        // cuInit(0);
        
        // TODO: Get allocation granularity
        // CUmemAllocationProp prop = {};
        // prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        // prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        // prop.location.id = 0;
        // cuMemGetAllocationGranularity(&granularity, &prop, 
        //                               CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        
        // TODO: Reserve virtual address range
        // reservedSize = ((maxSize + granularity - 1) / granularity) * granularity;
        // cuMemAddressReserve(&ptr, reservedSize, 0, 0, 0);
        
        // Placeholder: use cudaMalloc for now
        reservedSize = maxSize;
        mappedSize = 0;
        cudaMalloc((void**)&ptr, maxSize);
    }
    
    bool grow(size_t newSize) {
        // TODO: Implement VMM-based growth
        // 1. Round up to granularity
        // 2. Create physical allocation handle
        // 3. Map new physical memory to virtual range
        // 4. Set access permissions
        
        // Placeholder: just update size (memory already allocated)
        if (newSize <= reservedSize) {
            mappedSize = newSize;
            return true;
        }
        return false;
    }
    
    void* getPtr() { return (void*)ptr; }
    size_t getSize() { return mappedSize; }
    
    ~GrowableBuffer() {
        // TODO: Unmap, free physical memory, release virtual address
        cudaFree((void*)ptr);
    }
};

int main() {
    const size_t MAX_SIZE = 256 * 1024 * 1024;  // 256 MB max
    
    GrowableBuffer buffer(MAX_SIZE);
    
    // Start with 1M integers
    size_t size1 = 1000000 * sizeof(int);
    buffer.grow(size1);
    printf("Initial size: %zu bytes\n", buffer.getSize());
    
    // Fill and verify
    int* data = (int*)buffer.getPtr();
    fillBuffer<<<(1000000+255)/256, 256>>>(data, 1000000, 1);
    
    long long *d_sum;
    cudaMalloc(&d_sum, sizeof(long long));
    cudaMemset(d_sum, 0, sizeof(long long));
    sumBuffer<<<(1000000+255)/256, 256>>>(data, 1000000, d_sum);
    
    long long h_sum;
    cudaMemcpy(&h_sum, d_sum, sizeof(long long), cudaMemcpyDeviceToHost);
    printf("Sum after fill: %lld (expected 1000000)\n", h_sum);
    
    // Grow to 10M integers
    size_t size2 = 10000000 * sizeof(int);
    buffer.grow(size2);
    printf("After growth: %zu bytes\n", buffer.getSize());
    
    // Fill new region
    fillBuffer<<<(10000000+255)/256, 256>>>(data, 10000000, 1);
    cudaMemset(d_sum, 0, sizeof(long long));
    sumBuffer<<<(10000000+255)/256, 256>>>(data, 10000000, d_sum);
    cudaMemcpy(&h_sum, d_sum, sizeof(long long), cudaMemcpyDeviceToHost);
    printf("Sum after growth: %lld (expected 10000000)\n", h_sum);
    
    bool passed = (h_sum == 10000000);
    printf("Test %s\n", passed ? "PASSED" : "FAILED");
    
    cudaFree(d_sum);
    return passed ? 0 : 1;
}
