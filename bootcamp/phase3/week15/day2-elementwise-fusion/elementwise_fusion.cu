/**
 * elementwise_fusion.cu - Fuse element-wise operations
 * 
 * Learning objectives:
 * - Chain activation functions
 * - Fuse scale + bias + activation
 * - Measure bandwidth improvement
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// ============================================================================
// Unfused Kernels
// ============================================================================

__global__ void scale_kernel(const float* x, float scale, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = x[idx] * scale;
}

__global__ void bias_kernel(const float* x, const float* bias, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = x[idx] + bias[idx];
}

__global__ void relu_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = fmaxf(0.0f, x[idx]);
}

__global__ void sigmoid_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = 1.0f / (1.0f + expf(-x[idx]));
}

// ============================================================================
// Fused Kernel
// ============================================================================

__global__ void fused_scale_bias_relu_sigmoid(
    const float* x, float scale, const float* bias, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx] * scale + bias[idx];  // Scale + bias
        val = fmaxf(0.0f, val);                   // ReLU
        y[idx] = 1.0f / (1.0f + expf(-val));      // Sigmoid
    }
}

// ============================================================================
// GELU Fusion Example
// ============================================================================

// Unfused GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
__global__ void pow3_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = x[idx] * x[idx] * x[idx];
}

__global__ void gelu_inner_kernel(const float* x, const float* x3, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float sqrt_2_pi = 0.7978845608f;
        y[idx] = sqrt_2_pi * (x[idx] + 0.044715f * x3[idx]);
    }
}

__global__ void tanh_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = tanhf(x[idx]);
}

__global__ void gelu_outer_kernel(const float* x, const float* tanh_val, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = 0.5f * x[idx] * (1.0f + tanh_val[idx]);
    }
}

// Fused GELU
__global__ void fused_gelu(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float sqrt_2_pi = 0.7978845608f;
        float xi = x[idx];
        float inner = sqrt_2_pi * (xi + 0.044715f * xi * xi * xi);
        y[idx] = 0.5f * xi * (1.0f + tanhf(inner));
    }
}

int main() {
    printf("=== Element-wise Fusion Demo ===\n\n");
    
    const int N = 1 << 24;  // 16M elements
    const size_t bytes = N * sizeof(float);
    const float scale = 0.5f;
    
    // Allocate
    float *d_x, *d_bias;
    float *d_t1, *d_t2, *d_t3, *d_t4;  // Temporaries for unfused
    float *d_out_unfused, *d_out_fused;
    
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_bias, bytes);
    cudaMalloc(&d_t1, bytes);
    cudaMalloc(&d_t2, bytes);
    cudaMalloc(&d_t3, bytes);
    cudaMalloc(&d_t4, bytes);
    cudaMalloc(&d_out_unfused, bytes);
    cudaMalloc(&d_out_fused, bytes);
    
    // Initialize
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = 0.5f;
    cudaMemcpy(d_x, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_data, bytes, cudaMemcpyHostToDevice);
    delete[] h_data;
    
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ========================================================================
    // Test 1: Scale + Bias + ReLU + Sigmoid
    // ========================================================================
    {
        printf("1. Scale + Bias + ReLU + Sigmoid\n");
        printf("─────────────────────────────────────────\n");
        
        // Warmup
        scale_kernel<<<num_blocks, block_size>>>(d_x, scale, d_t1, N);
        cudaDeviceSynchronize();
        
        // Unfused
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            scale_kernel<<<num_blocks, block_size>>>(d_x, scale, d_t1, N);
            bias_kernel<<<num_blocks, block_size>>>(d_t1, d_bias, d_t2, N);
            relu_kernel<<<num_blocks, block_size>>>(d_t2, d_t3, N);
            sigmoid_kernel<<<num_blocks, block_size>>>(d_t3, d_out_unfused, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float unfused_ms;
        cudaEventElapsedTime(&unfused_ms, start, stop);
        unfused_ms /= 100;
        
        // Fused
        fused_scale_bias_relu_sigmoid<<<num_blocks, block_size>>>(
            d_x, scale, d_bias, d_out_fused, N);
        cudaDeviceSynchronize();
        
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            fused_scale_bias_relu_sigmoid<<<num_blocks, block_size>>>(
                d_x, scale, d_bias, d_out_fused, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float fused_ms;
        cudaEventElapsedTime(&fused_ms, start, stop);
        fused_ms /= 100;
        
        // Verify
        float h_unfused, h_fused;
        cudaMemcpy(&h_unfused, d_out_unfused, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_fused, d_out_fused, sizeof(float), cudaMemcpyDeviceToHost);
        
        // Memory analysis
        // Unfused: scale(2) + bias(3) + relu(2) + sigmoid(2) = 9 transactions
        // Fused: read x,bias, write y = 3 transactions
        
        float unfused_bytes = 9.0f * N * sizeof(float);
        float fused_bytes = 3.0f * N * sizeof(float);
        
        printf("   Unfused: 4 kernels, 9 memory transactions\n");
        printf("   Fused:   1 kernel,  3 memory transactions\n\n");
        printf("   Unfused: %.3f ms (%.0f GB/s)\n", 
               unfused_ms, unfused_bytes / (unfused_ms/1000) / 1e9);
        printf("   Fused:   %.3f ms (%.0f GB/s)\n",
               fused_ms, fused_bytes / (fused_ms/1000) / 1e9);
        printf("   Speedup: %.2fx\n", unfused_ms / fused_ms);
        printf("   Correct: %s (%.6f vs %.6f)\n\n", 
               fabsf(h_unfused - h_fused) < 1e-5 ? "YES" : "NO",
               h_unfused, h_fused);
    }
    
    // ========================================================================
    // Test 2: GELU Activation
    // ========================================================================
    {
        printf("2. GELU Activation (Complex Fusion)\n");
        printf("─────────────────────────────────────────\n");
        
        // Reset input
        float* h_input = new float[N];
        for (int i = 0; i < N; i++) h_input[i] = 0.1f * (i % 20 - 10);
        cudaMemcpy(d_x, h_input, bytes, cudaMemcpyHostToDevice);
        delete[] h_input;
        
        // Unfused GELU: x³, inner, tanh, outer
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            pow3_kernel<<<num_blocks, block_size>>>(d_x, d_t1, N);
            gelu_inner_kernel<<<num_blocks, block_size>>>(d_x, d_t1, d_t2, N);
            tanh_kernel<<<num_blocks, block_size>>>(d_t2, d_t3, N);
            gelu_outer_kernel<<<num_blocks, block_size>>>(d_x, d_t3, d_out_unfused, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float unfused_ms;
        cudaEventElapsedTime(&unfused_ms, start, stop);
        unfused_ms /= 100;
        
        // Fused GELU
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            fused_gelu<<<num_blocks, block_size>>>(d_x, d_out_fused, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float fused_ms;
        cudaEventElapsedTime(&fused_ms, start, stop);
        fused_ms /= 100;
        
        // Verify
        float h_unfused[5], h_fused[5];
        cudaMemcpy(h_unfused, d_out_unfused, 5 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fused, d_out_fused, 5 * sizeof(float), cudaMemcpyDeviceToHost);
        
        printf("   Unfused: 4 kernels (pow3, inner, tanh, outer)\n");
        printf("   Fused:   1 kernel\n\n");
        printf("   Unfused: %.3f ms\n", unfused_ms);
        printf("   Fused:   %.3f ms\n", fused_ms);
        printf("   Speedup: %.2fx\n\n", unfused_ms / fused_ms);
        
        bool correct = true;
        for (int i = 0; i < 5; i++) {
            if (fabsf(h_unfused[i] - h_fused[i]) > 1e-5) correct = false;
        }
        printf("   Verification: %s\n\n", correct ? "PASSED" : "FAILED");
    }
    
    printf("=== Key Points ===\n\n");
    printf("1. Consecutive element-wise ops should ALWAYS be fused\n");
    printf("2. Memory traffic reduction = number of intermediate arrays eliminated\n");
    printf("3. Complex activations (GELU, Swish) benefit greatly from fusion\n");
    printf("4. No register pressure concern for simple element-wise ops\n");
    
    // Cleanup
    cudaFree(d_x);
    cudaFree(d_bias);
    cudaFree(d_t1);
    cudaFree(d_t2);
    cudaFree(d_t3);
    cudaFree(d_t4);
    cudaFree(d_out_unfused);
    cudaFree(d_out_fused);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
