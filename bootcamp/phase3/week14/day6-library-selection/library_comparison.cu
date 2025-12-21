/**
 * library_comparison.cu - Compare library approaches
 * 
 * Learning objectives:
 * - Benchmark different approaches
 * - Understand when to use each library
 * - Make informed library choices
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <cstdio>
#include <cmath>

// ============================================================================
// Custom Implementations for Comparison
// ============================================================================

// Simple reduction kernel
__global__ void custom_reduce_kernel(const float* in, float* out, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

// Simple transform kernel
__global__ void custom_square_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * in[idx];
    }
}

// ============================================================================
// Timing Helper
// ============================================================================

float time_kernel(std::function<void()> kernel, int iterations = 10) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    kernel();
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms / iterations;
}

int main() {
    printf("=== Library Selection Comparison ===\n\n");
    
    const int N = 1 << 24;  // 16M elements
    
    // ========================================================================
    // Test 1: Reduction - Custom vs CUB vs Thrust
    // ========================================================================
    {
        printf("1. REDUCTION COMPARISON\n");
        printf("═══════════════════════════════════════════════════════════\n");
        printf("   %d elements (%.0f MB)\n\n", N, N * sizeof(float) / 1e6);
        
        // Allocate
        float *d_in, *d_out;
        cudaMalloc(&d_in, N * sizeof(float));
        cudaMalloc(&d_out, sizeof(float));
        
        thrust::device_vector<float> d_thrust(N);
        thrust::fill(d_thrust.begin(), d_thrust.end(), 1.0f);
        
        cudaMemcpy(d_in, thrust::raw_pointer_cast(d_thrust.data()), 
                   N * sizeof(float), cudaMemcpyDeviceToDevice);
        
        // CUB setup
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, N);
        cudaMalloc(&d_temp, temp_bytes);
        
        // Benchmark Custom
        float custom_time = time_kernel([&]() {
            cudaMemset(d_out, 0, sizeof(float));
            int block_size = 256;
            int num_blocks = (N + block_size - 1) / block_size;
            custom_reduce_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
                d_in, d_out, N);
        });
        
        // Benchmark CUB
        float cub_time = time_kernel([&]() {
            cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, N);
        });
        
        // Benchmark Thrust
        float thrust_time = time_kernel([&]() {
            thrust::reduce(d_thrust.begin(), d_thrust.end());
        });
        
        printf("   Method          Time (ms)    Speedup vs Custom\n");
        printf("   ─────────────   ─────────    ─────────────────\n");
        printf("   Custom          %8.3f     1.00x\n", custom_time);
        printf("   CUB             %8.3f     %.2fx\n", cub_time, custom_time / cub_time);
        printf("   Thrust          %8.3f     %.2fx\n\n", thrust_time, custom_time / thrust_time);
        
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_temp);
    }
    
    // ========================================================================
    // Test 2: Sort - CUB vs Thrust
    // ========================================================================
    {
        printf("2. SORT COMPARISON\n");
        printf("═══════════════════════════════════════════════════════════\n");
        printf("   %d elements\n\n", N);
        
        // Generate random data
        thrust::host_vector<int> h_data(N);
        for (int i = 0; i < N; i++) h_data[i] = rand();
        
        // CUB setup
        int *d_in, *d_out;
        cudaMalloc(&d_in, N * sizeof(int));
        cudaMalloc(&d_out, N * sizeof(int));
        
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, N);
        cudaMalloc(&d_temp, temp_bytes);
        
        // Benchmark CUB
        float cub_time = time_kernel([&]() {
            cudaMemcpy(d_in, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice);
            cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, N);
        }, 5);
        
        // Benchmark Thrust
        thrust::device_vector<int> d_thrust(N);
        float thrust_time = time_kernel([&]() {
            d_thrust = h_data;  // Copy
            thrust::sort(d_thrust.begin(), d_thrust.end());
        }, 5);
        
        printf("   Method          Time (ms)    Elements/sec\n");
        printf("   ─────────────   ─────────    ────────────\n");
        printf("   CUB             %8.2f     %.2e\n", cub_time, N / (cub_time / 1000));
        printf("   Thrust          %8.2f     %.2e\n\n", thrust_time, N / (thrust_time / 1000));
        
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_temp);
    }
    
    // ========================================================================
    // Test 3: Transform - Custom vs Thrust
    // ========================================================================
    {
        printf("3. TRANSFORM COMPARISON\n");
        printf("═══════════════════════════════════════════════════════════\n");
        printf("   Square %d elements\n\n", N);
        
        float *d_in, *d_out;
        cudaMalloc(&d_in, N * sizeof(float));
        cudaMalloc(&d_out, N * sizeof(float));
        
        thrust::device_vector<float> d_thrust_in(N, 2.0f);
        thrust::device_vector<float> d_thrust_out(N);
        cudaMemcpy(d_in, thrust::raw_pointer_cast(d_thrust_in.data()),
                   N * sizeof(float), cudaMemcpyDeviceToDevice);
        
        // Benchmark Custom
        int block_size = 256;
        int num_blocks = (N + block_size - 1) / block_size;
        float custom_time = time_kernel([&]() {
            custom_square_kernel<<<num_blocks, block_size>>>(d_in, d_out, N);
        });
        
        // Benchmark Thrust
        float thrust_time = time_kernel([&]() {
            thrust::transform(d_thrust_in.begin(), d_thrust_in.end(),
                              d_thrust_out.begin(),
                              [] __device__ (float x) { return x * x; });
        });
        
        float gbps_custom = 2.0f * N * sizeof(float) / (custom_time / 1000) / 1e9;
        float gbps_thrust = 2.0f * N * sizeof(float) / (thrust_time / 1000) / 1e9;
        
        printf("   Method          Time (ms)    Bandwidth (GB/s)\n");
        printf("   ─────────────   ─────────    ────────────────\n");
        printf("   Custom          %8.3f     %.1f\n", custom_time, gbps_custom);
        printf("   Thrust          %8.3f     %.1f\n\n", thrust_time, gbps_thrust);
        
        cudaFree(d_in);
        cudaFree(d_out);
    }
    
    // ========================================================================
    // Decision Summary
    // ========================================================================
    printf("═══════════════════════════════════════════════════════════\n");
    printf("DECISION GUIDE\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    
    printf("┌─────────────────┬───────────────────────────────────────┐\n");
    printf("│ Use Case        │ Recommendation                        │\n");
    printf("├─────────────────┼───────────────────────────────────────┤\n");
    printf("│ BLAS operations │ cuBLAS (vendor-tuned)                 │\n");
    printf("│ Reduction       │ CUB (best perf) or Thrust (easy)      │\n");
    printf("│ Sort            │ CUB or Thrust (similar performance)   │\n");
    printf("│ Simple transform│ Custom kernel (minimal overhead)      │\n");
    printf("│ Complex pipeline│ Thrust (readable, maintainable)       │\n");
    printf("│ Inside kernels  │ CUB block/warp primitives             │\n");
    printf("│ Prototyping     │ Thrust (fastest development)          │\n");
    printf("│ Production      │ Profile and optimize critical paths   │\n");
    printf("└─────────────────┴───────────────────────────────────────┘\n\n");
    
    printf("KEY INSIGHTS:\n");
    printf("  • CUB generally fastest for primitives\n");
    printf("  • Thrust adds ~5-10%% overhead but much easier\n");
    printf("  • Custom kernels win for simple ops (no library overhead)\n");
    printf("  • Vendor libraries (cuBLAS, cuDNN) hard to beat\n");
    printf("  • Profile YOUR use case - results vary!\n\n");
    
    return 0;
}
