/**
 * spill_demo.cu - Demonstrate register spilling to local memory
 * 
 * Learning objectives:
 * - See performance impact of spilling
 * - Recognize spill conditions
 * - Strategies to reduce spilling
 */

#include <cuda_runtime.h>
#include <cstdio>

// No spilling - small number of variables
__global__ void no_spill(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float a = data[idx];
    float b = a * 2.0f;
    float c = sinf(a);
    float d = cosf(b);
    
    data[idx] = a + b + c + d;
}

// Likely spilling - large local array
__global__ void likely_spill(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // This array is too big for registers → spills to local memory
    float local_array[128];  // 128 floats = 512 bytes per thread!
    
    // Initialize
    for (int i = 0; i < 128; i++) {
        local_array[i] = data[idx] * (float)i;
    }
    
    // Compute
    float sum = 0.0f;
    for (int i = 0; i < 128; i++) {
        sum += local_array[i];
    }
    
    data[idx] = sum;
}

// Forced spilling - many independent variables
__global__ void many_variables(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Many independent variables that compiler can't optimize away
    float v0 = data[idx] * 1.0f;
    float v1 = sinf(v0);
    float v2 = cosf(v0);
    float v3 = tanf(v0);
    float v4 = expf(v0 * 0.01f);
    float v5 = logf(fabsf(v0) + 1.0f);
    float v6 = sqrtf(fabsf(v0) + 1.0f);
    float v7 = v0 + v1;
    float v8 = v2 + v3;
    float v9 = v4 + v5;
    float v10 = v6 + v7;
    float v11 = v8 + v9;
    float v12 = v10 + v11;
    float v13 = sinf(v12);
    float v14 = cosf(v13);
    float v15 = tanf(v14);
    float v16 = v0 * v1;
    float v17 = v2 * v3;
    float v18 = v4 * v5;
    float v19 = v6 * v7;
    float v20 = v8 * v9;
    float v21 = v10 * v11;
    float v22 = v12 * v13;
    float v23 = v14 * v15;
    
    // Use all variables to prevent optimization
    data[idx] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 +
                v10 + v11 + v12 + v13 + v14 + v15 + v16 + v17 + v18 + v19 +
                v20 + v21 + v22 + v23;
}

// Reusing variables - reduce register pressure
__global__ void reuse_variables(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float val = data[idx];
    float sum = 0.0f;
    
    // Reuse 'val' for intermediate computations
    val = sinf(val); sum += val;
    val = cosf(val); sum += val;
    val = tanf(val); sum += val;
    val = expf(val * 0.01f); sum += val;
    val = logf(fabsf(val) + 1.0f); sum += val;
    val = sqrtf(fabsf(val) + 1.0f); sum += val;
    
    data[idx] = sum;
}

int main() {
    printf("=== Register Spilling Demo ===\n\n");
    
    printf("Register spilling occurs when:\n");
    printf("1. Too many local variables\n");
    printf("2. Large local arrays\n");
    printf("3. Complex expressions\n\n");
    
    const int N = 1 << 20;
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    // Warmup
    no_spill<<<num_blocks, block_size>>>(d_data, N);
    cudaDeviceSynchronize();
    
    printf("=== Benchmark ===\n\n");
    
    // No spill
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        no_spill<<<num_blocks, block_size>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("No spill (few vars):     %8.3f ms\n", ms / 100);
    float baseline = ms / 100;
    
    // Reuse variables
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        reuse_variables<<<num_blocks, block_size>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Reuse variables:         %8.3f ms (%.1fx)\n", ms / 100, (ms / 100) / baseline);
    
    // Many variables
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        many_variables<<<num_blocks, block_size>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Many variables:          %8.3f ms (%.1fx)\n", ms / 100, (ms / 100) / baseline);
    
    // Large local array
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        likely_spill<<<num_blocks, block_size>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Large local array:       %8.3f ms (%.1fx)\n", ms / 100, (ms / 100) / baseline);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    
    printf("\n=== Detecting Spills ===\n");
    printf("Compile with: nvcc --ptxas-options=-v\n");
    printf("Look for: 'lmem' (local memory) usage > 0\n");
    printf("\n");
    printf("Example output:\n");
    printf("  Used 32 registers, 512 bytes lmem  ← Spilling!\n");
    printf("  Used 24 registers, 0 bytes lmem    ← No spilling\n");
    
    printf("\n=== Reducing Spills ===\n");
    printf("1. Reuse variables instead of creating new ones\n");
    printf("2. Use shared memory for large arrays\n");
    printf("3. Split kernel into multiple passes\n");
    printf("4. Use __launch_bounds__ to let compiler optimize\n");
    
    return 0;
}
