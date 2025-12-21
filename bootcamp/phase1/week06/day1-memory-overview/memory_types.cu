/**
 * memory_types.cu - Demonstrate CUDA memory types
 * 
 * Learning objectives:
 * - Understand memory hierarchy
 * - See how to declare each memory type
 * - Observe access patterns
 */

#include <cuda_runtime.h>
#include <cstdio>

// Constant memory (device-wide, read-only)
__constant__ float d_constants[256];

// Global memory kernel
__global__ void global_memory_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;  // Read and write global memory
    }
}

// Shared memory kernel
__global__ void shared_memory_kernel(float* data, int n) {
    __shared__ float smem[256];  // Shared within block
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load from global to shared
    if (idx < n) {
        smem[tid] = data[idx];
    }
    __syncthreads();
    
    // Work with shared memory (reverse within block)
    if (idx < n) {
        data[idx] = smem[blockDim.x - 1 - tid];
    }
}

// Constant memory kernel
__global__ void constant_memory_kernel(float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // All threads read same constant - broadcast!
        output[idx] = d_constants[0] * idx;
    }
}

// Register-heavy kernel (to show register usage)
__global__ void register_heavy_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Many local variables → register usage
    float a = data[idx];
    float b = a * 2.0f;
    float c = sinf(a);
    float d = cosf(b);
    float e = a + b + c + d;
    float f = sqrtf(fabsf(e));
    float g = expf(f * 0.01f);
    float h = logf(g + 1.0f);
    
    data[idx] = a + b + c + d + e + f + g + h;
}

// Local memory kernel (arrays too big for registers)
__global__ void local_memory_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // This array is too big for registers → goes to local memory (slow!)
    float local_array[64];  // 64 * 4 bytes = 256 bytes per thread
    
    // Initialize local array
    for (int i = 0; i < 64; i++) {
        local_array[i] = data[idx] * i;
    }
    
    // Compute with local array
    float sum = 0.0f;
    for (int i = 0; i < 64; i++) {
        sum += local_array[i];
    }
    
    data[idx] = sum;
}

int main() {
    printf("=== CUDA Memory Types Demo ===\n\n");
    
    // Get device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device: %s\n", prop.name);
    printf("Global memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Shared memory per SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Constant memory: 64 KB (fixed)\n\n");
    
    const int N = 256;
    float* d_data;
    float* h_data = new float[N];
    
    for (int i = 0; i < N; i++) h_data[i] = (float)i;
    cudaMalloc(&d_data, N * sizeof(float));
    
    // Demo 1: Global memory
    printf("=== Demo 1: Global Memory ===\n");
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    global_memory_kernel<<<1, N>>>(d_data, N);
    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Global memory: data[0] = %.1f (was 0, now doubled)\n\n", h_data[0]);
    
    // Demo 2: Shared memory
    printf("=== Demo 2: Shared Memory ===\n");
    for (int i = 0; i < N; i++) h_data[i] = (float)i;
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    shared_memory_kernel<<<1, N>>>(d_data, N);
    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Shared memory: data[0] = %.1f (reversed from 255)\n\n", h_data[0]);
    
    // Demo 3: Constant memory
    printf("=== Demo 3: Constant Memory ===\n");
    float h_constants[256];
    h_constants[0] = 2.0f;  // Multiplier
    cudaMemcpyToSymbol(d_constants, h_constants, sizeof(float));
    
    constant_memory_kernel<<<1, N>>>(d_data, N);
    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Constant memory: data[10] = %.1f (10 * 2.0)\n\n", h_data[10]);
    
    // Demo 4: Show register usage with compilation
    printf("=== Demo 4: Register Usage ===\n");
    printf("Compile with: nvcc --ptxas-options=-v to see register usage\n");
    printf("High register usage → fewer concurrent threads\n\n");
    
    register_heavy_kernel<<<1, N>>>(d_data, N);
    cudaDeviceSynchronize();
    
    // Demo 5: Local memory warning
    printf("=== Demo 5: Local Memory (Slow!) ===\n");
    printf("Large local arrays spill to local memory (backed by global)\n");
    printf("64-element float array per thread = 256 bytes per thread\n\n");
    
    local_memory_kernel<<<1, N>>>(d_data, N);
    cudaDeviceSynchronize();
    
    cudaFree(d_data);
    delete[] h_data;
    
    printf("=== Memory Type Summary ===\n");
    printf("Registers:  Fastest, per-thread, limited (~255)\n");
    printf("Shared:     Fast, per-block, programmer-managed (48-164 KB/SM)\n");
    printf("L1/L2:      Automatic caching of global memory\n");
    printf("Global:     Large but slow, coalescing matters\n");
    printf("Constant:   Read-only, broadcast optimized (64 KB)\n");
    printf("Local:      Per-thread in global memory (avoid!)\n");
    
    return 0;
}
