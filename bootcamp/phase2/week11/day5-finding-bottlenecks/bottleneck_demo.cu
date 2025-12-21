/**
 * bottleneck_demo.cu - Kernels with identifiable bottlenecks
 * 
 * Profile with: ncu --set full ./build/bottleneck_demo
 * Look at the "Speed of Light" and bottleneck analysis sections
 */

#include <cuda_runtime.h>
#include <cstdio>

// ========== Bottleneck 1: Memory bandwidth limited ==========
__global__ void bottleneck_bandwidth(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple copy - pure bandwidth
        out[idx] = in[idx];
    }
}

// ========== Bottleneck 2: Memory latency (uncoalesced) ==========
__global__ void bottleneck_latency(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Strided access - can't coalesce, exposed latency
    int strided_idx = (idx * 32) % n;
    if (idx < n / 32) {
        out[idx] = in[strided_idx];
    }
}

// ========== Bottleneck 3: Low occupancy ==========
__global__ __launch_bounds__(64, 2)  // Force low occupancy
void bottleneck_occupancy(float* out, int n) {
    __shared__ float smem[8192];  // 32KB - limits blocks
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Fill shared memory
    for (int i = tid; i < 8192; i += blockDim.x) {
        smem[i] = (float)i;
    }
    __syncthreads();
    
    if (idx < n) {
        float sum = 0;
        for (int i = 0; i < 64; i++) {
            sum += smem[(tid + i) % 8192];
        }
        out[idx] = sum;
    }
}

// ========== Bottleneck 4: Warp divergence ==========
__global__ void bottleneck_divergence(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = 1.0f;
        
        // Every thread takes different number of iterations
        int iters = idx % 32;  // 0-31 iterations per lane
        for (int i = 0; i < iters; i++) {
            val = val * 1.01f + 0.01f;
        }
        out[idx] = val;
    }
}

// ========== Bottleneck 5: Instruction dependencies ==========
__global__ void bottleneck_dependencies(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = 1.0f;
        
        // Long serial chain - each depends on previous
        for (int i = 0; i < 200; i++) {
            val = val * 1.0001f;  // RAW dependency
        }
        out[idx] = val;
    }
}

// ========== Bottleneck 6: Atomic contention ==========
__global__ void bottleneck_atomics(int* counter, int n) {
    // All threads increment same counter - massive contention
    atomicAdd(counter, 1);
}

// ========== Reference: Well-optimized kernel ==========
__global__ void optimized_reference(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        
        // Independent chains for ILP
        float v0 = val, v1 = val, v2 = val, v3 = val;
        for (int i = 0; i < 25; i++) {
            v0 = v0 * 1.0001f + 0.0001f;
            v1 = v1 * 1.0002f + 0.0002f;
            v2 = v2 * 1.0003f + 0.0003f;
            v3 = v3 * 1.0004f + 0.0004f;
        }
        out[idx] = v0 + v1 + v2 + v3;
    }
}

int main() {
    printf("Bottleneck Identification Demo\n\n");
    printf("Profile with: ncu --set full ./build/bottleneck_demo\n\n");
    
    const int N = 1 << 22;
    float *d_in, *d_out;
    int *d_counter;
    
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));
    
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    cudaMemcpy(d_in, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    printf("Launching kernels with different bottlenecks...\n\n");
    
    printf("1. bottleneck_bandwidth - memory BW limited\n");
    printf("   NCU shows: High DRAM throughput, low compute\n");
    bottleneck_bandwidth<<<blocks, threads>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    
    printf("2. bottleneck_latency - uncoalesced access\n");
    printf("   NCU shows: Low memory efficiency, high sectors/request\n");
    bottleneck_latency<<<blocks/32, threads>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    
    printf("3. bottleneck_occupancy - resource limited\n");
    printf("   NCU shows: Low achieved occupancy\n");
    bottleneck_occupancy<<<N/64, 64>>>(d_out, N);
    cudaDeviceSynchronize();
    
    printf("4. bottleneck_divergence - branch divergence\n");
    printf("   NCU shows: Low branch efficiency, predicated instructions\n");
    bottleneck_divergence<<<blocks, threads>>>(d_out, N);
    cudaDeviceSynchronize();
    
    printf("5. bottleneck_dependencies - serial dependencies\n");
    printf("   NCU shows: Short scoreboard stalls\n");
    bottleneck_dependencies<<<blocks, threads>>>(d_out, N);
    cudaDeviceSynchronize();
    
    printf("6. bottleneck_atomics - atomic contention\n");
    printf("   NCU shows: L2 atomic throughput, long tail\n");
    bottleneck_atomics<<<1024, 256>>>(d_counter, N);
    cudaDeviceSynchronize();
    
    printf("7. optimized_reference - balanced, efficient\n");
    printf("   NCU shows: Good utilization of both memory and compute\n");
    optimized_reference<<<blocks, threads>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    
    printf("\n=== Bottleneck Diagnosis Checklist ===\n\n");
    printf("1. Check 'Speed of Light' - Memory vs Compute bound?\n");
    printf("2. Memory-bound?\n");
    printf("   - Check load/store efficiency\n");
    printf("   - Check L1/L2 hit rates\n");
    printf("   - Check sectors per request\n");
    printf("3. Compute-bound?\n");
    printf("   - Check occupancy\n");
    printf("   - Check stall reasons\n");
    printf("   - Check instruction mix\n");
    printf("4. Neither fully utilized?\n");
    printf("   - Check for latency issues\n");
    printf("   - Check for synchronization\n");
    printf("   - Check for launch overhead\n");
    
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_counter);
    delete[] h_data;
    
    return 0;
}
