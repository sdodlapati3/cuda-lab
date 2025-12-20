#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void fillBuffer(int* data, int n, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

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

class GrowableBuffer {
    CUdeviceptr ptr;
    size_t reservedSize;
    size_t mappedSize;
    size_t granularity;
    CUmemAllocationProp prop;
    CUmemAccessDesc accessDesc;
    std::vector<CUmemGenericAllocationHandle> handles;
    
public:
    GrowableBuffer(size_t maxSize) : mappedSize(0) {
        cuInit(0);
        
        int device;
        cudaGetDevice(&device);
        
        memset(&prop, 0, sizeof(prop));
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        
        cuMemGetAllocationGranularity(&granularity, &prop, 
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        
        reservedSize = ((maxSize + granularity - 1) / granularity) * granularity;
        cuMemAddressReserve(&ptr, reservedSize, granularity, 0, 0);
        
        memset(&accessDesc, 0, sizeof(accessDesc));
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = device;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    
    bool grow(size_t newSize) {
        if (newSize <= mappedSize) return true;
        if (newSize > reservedSize) return false;
        
        size_t alignedNew = ((newSize + granularity - 1) / granularity) * granularity;
        size_t toMap = alignedNew - mappedSize;
        
        CUmemGenericAllocationHandle handle;
        CUresult res = cuMemCreate(&handle, toMap, &prop, 0);
        if (res != CUDA_SUCCESS) return false;
        
        res = cuMemMap(ptr + mappedSize, toMap, 0, handle, 0);
        if (res != CUDA_SUCCESS) {
            cuMemRelease(handle);
            return false;
        }
        
        res = cuMemSetAccess(ptr + mappedSize, toMap, &accessDesc, 1);
        if (res != CUDA_SUCCESS) return false;
        
        handles.push_back(handle);
        mappedSize = alignedNew;
        return true;
    }
    
    void* getPtr() { return (void*)ptr; }
    size_t getSize() { return mappedSize; }
    
    ~GrowableBuffer() {
        cuMemUnmap(ptr, mappedSize);
        for (auto& h : handles) {
            cuMemRelease(h);
        }
        cuMemAddressFree(ptr, reservedSize);
    }
};

int main() {
    const size_t MAX_SIZE = 256 * 1024 * 1024;
    
    GrowableBuffer buffer(MAX_SIZE);
    
    size_t size1 = 1000000 * sizeof(int);
    buffer.grow(size1);
    printf("Initial size: %zu bytes\n", buffer.getSize());
    
    int* data = (int*)buffer.getPtr();
    fillBuffer<<<(1000000+255)/256, 256>>>(data, 1000000, 1);
    
    long long *d_sum;
    cudaMalloc(&d_sum, sizeof(long long));
    cudaMemset(d_sum, 0, sizeof(long long));
    sumBuffer<<<(1000000+255)/256, 256>>>(data, 1000000, d_sum);
    
    long long h_sum;
    cudaMemcpy(&h_sum, d_sum, sizeof(long long), cudaMemcpyDeviceToHost);
    printf("Sum after fill: %lld (expected 1000000)\n", h_sum);
    
    size_t size2 = 10000000 * sizeof(int);
    buffer.grow(size2);
    printf("After growth: %zu bytes\n", buffer.getSize());
    
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
