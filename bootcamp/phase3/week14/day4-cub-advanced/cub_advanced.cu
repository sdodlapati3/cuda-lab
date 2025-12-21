/**
 * cub_advanced.cu - Block/Warp-level CUB and custom operators
 * 
 * Learning objectives:
 * - Use BlockReduce, BlockScan in kernels
 * - Use WarpReduce for warp-level ops
 * - Create custom reduction operators
 */

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cfloat>

// ============================================================================
// Custom Reduction Operators
// ============================================================================

struct MinMaxOp {
    __device__ __forceinline__
    cub::KeyValuePair<float, float> operator()(
        const cub::KeyValuePair<float, float>& a,
        const cub::KeyValuePair<float, float>& b) const {
        return cub::KeyValuePair<float, float>(
            (a.key < b.key) ? a.key : b.key,    // min
            (a.value > b.value) ? a.value : b.value  // max
        );
    }
};

// ============================================================================
// Block-Level Reduction Kernel
// ============================================================================

template<int BLOCK_SIZE>
__global__ void block_reduce_kernel(const float* in, float* block_sums, int n) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? in[idx] : 0.0f;
    
    float block_sum = BlockReduce(temp_storage).Sum(val);
    
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = block_sum;
    }
}

// ============================================================================
// Block-Level Scan Kernel
// ============================================================================

template<int BLOCK_SIZE>
__global__ void block_scan_kernel(const int* in, int* out, int n) {
    typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (idx < n) ? in[idx] : 0;
    
    int result;
    BlockScan(temp_storage).ExclusiveSum(val, result);
    
    if (idx < n) {
        out[idx] = result;
    }
}

// ============================================================================
// Warp-Level Reduction
// ============================================================================

__global__ void warp_reduce_kernel(const float* in, float* warp_sums, int n) {
    typedef cub::WarpReduce<float> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[8];  // 256/32 warps
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = threadIdx.x / 32;
    
    float val = (idx < n) ? in[idx] : 0.0f;
    
    float warp_sum = WarpReduce(temp_storage[warp_id]).Sum(val);
    
    // First lane of each warp writes
    if ((threadIdx.x % 32) == 0) {
        int global_warp = blockIdx.x * (blockDim.x / 32) + warp_id;
        warp_sums[global_warp] = warp_sum;
    }
}

// ============================================================================
// Custom Operator: ArgMax
// ============================================================================

struct ArgMax {
    __device__ __forceinline__
    cub::KeyValuePair<int, float> operator()(
        const cub::KeyValuePair<int, float>& a,
        const cub::KeyValuePair<int, float>& b) const {
        return (b.value > a.value) ? b : a;
    }
};

template<int BLOCK_SIZE>
__global__ void argmax_kernel(const float* in, int* max_idx, float* max_val, int n) {
    typedef cub::BlockReduce<cub::KeyValuePair<int, float>, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Create key-value pair (index, value)
    cub::KeyValuePair<int, float> kv;
    kv.key = idx;
    kv.value = (idx < n) ? in[idx] : -FLT_MAX;
    
    // Reduce with ArgMax
    auto result = BlockReduce(temp_storage).Reduce(kv, ArgMax());
    
    if (threadIdx.x == 0) {
        // Atomic compare-and-swap for global max
        // Simplified: just write block results
        max_idx[blockIdx.x] = result.key;
        max_val[blockIdx.x] = result.value;
    }
}

// ============================================================================
// Block Load/Store with CUB
// ============================================================================

template<int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void vectorized_transform(const float* in, float* out, int n) {
    typedef cub::BlockLoad<float, BLOCK_SIZE, ITEMS_PER_THREAD, 
                           cub::BLOCK_LOAD_VECTORIZE> BlockLoad;
    typedef cub::BlockStore<float, BLOCK_SIZE, ITEMS_PER_THREAD,
                            cub::BLOCK_STORE_VECTORIZE> BlockStore;
    
    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockStore::TempStorage store;
    } temp_storage;
    
    float items[ITEMS_PER_THREAD];
    
    int block_offset = blockIdx.x * BLOCK_SIZE * ITEMS_PER_THREAD;
    
    // Vectorized load
    BlockLoad(temp_storage.load).Load(in + block_offset, items);
    
    __syncthreads();
    
    // Transform
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        items[i] = items[i] * 2.0f + 1.0f;
    }
    
    // Vectorized store
    BlockStore(temp_storage.store).Store(out + block_offset, items);
}

int main() {
    printf("=== CUB Advanced Demo ===\n\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ========================================================================
    // Part 1: Block-Level Reduce
    // ========================================================================
    {
        printf("1. BlockReduce\n");
        printf("─────────────────────────────────────────\n");
        
        const int N = 1 << 20;
        const int BLOCK_SIZE = 256;
        const int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        float* h_in = new float[N];
        for (int i = 0; i < N; i++) h_in[i] = 1.0f;
        
        float *d_in, *d_block_sums;
        cudaMalloc(&d_in, N * sizeof(float));
        cudaMalloc(&d_block_sums, num_blocks * sizeof(float));
        cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
        
        block_reduce_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(
            d_in, d_block_sums, N);
        
        float* h_block_sums = new float[num_blocks];
        cudaMemcpy(h_block_sums, d_block_sums, num_blocks * sizeof(float), 
                   cudaMemcpyDeviceToHost);
        
        float total = 0;
        for (int i = 0; i < num_blocks; i++) total += h_block_sums[i];
        
        printf("   %d blocks, each reduces %d elements\n", num_blocks, BLOCK_SIZE);
        printf("   Total sum: %.0f (expected %d)\n\n", total, N);
        
        cudaFree(d_in);
        cudaFree(d_block_sums);
        delete[] h_in;
        delete[] h_block_sums;
    }
    
    // ========================================================================
    // Part 2: Warp-Level Reduce
    // ========================================================================
    {
        printf("2. WarpReduce\n");
        printf("─────────────────────────────────────────\n");
        
        const int N = 256;
        const int num_warps = N / 32;
        
        float* h_in = new float[N];
        for (int i = 0; i < N; i++) h_in[i] = 1.0f;
        
        float *d_in, *d_warp_sums;
        cudaMalloc(&d_in, N * sizeof(float));
        cudaMalloc(&d_warp_sums, num_warps * sizeof(float));
        cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
        
        warp_reduce_kernel<<<1, 256>>>(d_in, d_warp_sums, N);
        
        float h_warp_sums[8];
        cudaMemcpy(h_warp_sums, d_warp_sums, num_warps * sizeof(float), 
                   cudaMemcpyDeviceToHost);
        
        printf("   %d warps, each sums 32 ones\n", num_warps);
        printf("   Warp sums: ");
        for (int i = 0; i < num_warps; i++) printf("%.0f ", h_warp_sums[i]);
        printf("\n\n");
        
        cudaFree(d_in);
        cudaFree(d_warp_sums);
        delete[] h_in;
    }
    
    // ========================================================================
    // Part 3: ArgMax with Custom Operator
    // ========================================================================
    {
        printf("3. Custom Operator: ArgMax\n");
        printf("─────────────────────────────────────────\n");
        
        const int N = 1024;
        
        float* h_in = new float[N];
        int expected_idx = 0;
        float expected_val = -FLT_MAX;
        for (int i = 0; i < N; i++) {
            h_in[i] = (float)(rand() % 1000);
            if (h_in[i] > expected_val) {
                expected_val = h_in[i];
                expected_idx = i;
            }
        }
        
        float *d_in;
        int *d_max_idx;
        float *d_max_val;
        cudaMalloc(&d_in, N * sizeof(float));
        cudaMalloc(&d_max_idx, sizeof(int));
        cudaMalloc(&d_max_val, sizeof(float));
        cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
        
        argmax_kernel<1024><<<1, 1024>>>(d_in, d_max_idx, d_max_val, N);
        
        int result_idx;
        float result_val;
        cudaMemcpy(&result_idx, d_max_idx, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&result_val, d_max_val, sizeof(float), cudaMemcpyDeviceToHost);
        
        printf("   Found max: value=%.0f at index=%d\n", result_val, result_idx);
        printf("   Expected:  value=%.0f at index=%d\n\n", expected_val, expected_idx);
        
        cudaFree(d_in);
        cudaFree(d_max_idx);
        cudaFree(d_max_val);
        delete[] h_in;
    }
    
    // ========================================================================
    // Part 4: Vectorized Load/Store
    // ========================================================================
    {
        printf("4. BlockLoad/BlockStore (Vectorized)\n");
        printf("─────────────────────────────────────────\n");
        
        const int N = 1 << 20;
        const int BLOCK_SIZE = 256;
        const int ITEMS_PER_THREAD = 4;
        const int TILE_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD;
        const int num_blocks = N / TILE_SIZE;
        
        float *d_in, *d_out;
        cudaMalloc(&d_in, N * sizeof(float));
        cudaMalloc(&d_out, N * sizeof(float));
        
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            vectorized_transform<BLOCK_SIZE, ITEMS_PER_THREAD>
                <<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        float gbps = 2.0f * N * sizeof(float) / (ms / 100 / 1000) / 1e9;
        
        printf("   %d elements, %d items per thread\n", N, ITEMS_PER_THREAD);
        printf("   Bandwidth: %.2f GB/s\n\n", gbps);
        
        cudaFree(d_in);
        cudaFree(d_out);
    }
    
    printf("=== Key Points ===\n\n");
    printf("1. BlockReduce/BlockScan for cooperative block ops\n");
    printf("2. WarpReduce for warp-level ops\n");
    printf("3. Custom operators via functors\n");
    printf("4. BlockLoad/Store for optimized memory access\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
