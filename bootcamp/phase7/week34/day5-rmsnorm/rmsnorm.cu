/**
 * Week 34, Day 5: RMSNorm
 * 
 * RMSNorm(x) = x / sqrt(mean(x²) + eps) * gamma
 * 
 * Simpler than LayerNorm: no mean subtraction, no beta.
 * Used in LLaMA, Gemma, and other modern LLMs.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define BLOCK_SIZE 256

__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = warpSize/2; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void rmsNormKernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    float* __restrict__ output,
    int hidden, float eps
) {
    __shared__ float s_rms_inv;
    
    int batch_idx = blockIdx.x;
    const float* x = input + batch_idx * hidden;
    float* y = output + batch_idx * hidden;
    
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float val = x[i];
        sum_sq += val * val;
    }
    
    sum_sq = blockReduceSum(sum_sq);
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / hidden + eps);
        s_rms_inv = 1.0f / rms;
    }
    __syncthreads();
    
    float rms_inv = s_rms_inv;
    
    // Normalize
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        y[i] = x[i] * rms_inv * gamma[i];
    }
}

// CPU reference
void rmsNormCPU(const float* input, const float* gamma, float* output,
                int batch, int hidden, float eps) {
    for (int b = 0; b < batch; b++) {
        const float* x = input + b * hidden;
        float* y = output + b * hidden;
        
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden; i++) sum_sq += x[i] * x[i];
        float rms_inv = 1.0f / sqrtf(sum_sq / hidden + eps);
        
        for (int i = 0; i < hidden; i++) {
            y[i] = x[i] * rms_inv * gamma[i];
        }
    }
}

int main() {
    printf("Week 34 Day 5: RMSNorm\n\n");
    
    printf("RMSNorm Formula:\n");
    printf("  RMSNorm(x) = x / sqrt(mean(x²) + eps) * gamma\n\n");
    
    printf("Comparison with LayerNorm:\n");
    printf("┌─────────────┬────────────────────────────────────┐\n");
    printf("│ LayerNorm   │ (x - mean) / sqrt(var + eps) * γ + β │\n");
    printf("│ RMSNorm     │ x / sqrt(mean(x²) + eps) * γ       │\n");
    printf("├─────────────┼────────────────────────────────────┤\n");
    printf("│ Params      │ LayerNorm: 2n, RMSNorm: n          │\n");
    printf("│ Compute     │ RMSNorm ~15%% faster                │\n");
    printf("│ Quality     │ Similar for LLMs                   │\n");
    printf("└─────────────┴────────────────────────────────────┘\n\n");
    
    const int batch = 64, hidden = 4096;
    const float eps = 1e-6f;
    
    float *h_input = new float[batch * hidden];
    float *h_gamma = new float[hidden];
    float *h_output_cpu = new float[batch * hidden];
    float *h_output_gpu = new float[batch * hidden];
    
    for (int i = 0; i < batch * hidden; i++) h_input[i] = (float)(rand() % 100) / 50.0f - 1.0f;
    for (int i = 0; i < hidden; i++) h_gamma[i] = 1.0f;
    
    // CPU
    rmsNormCPU(h_input, h_gamma, h_output_cpu, batch, hidden, eps);
    
    // GPU
    float *d_input, *d_gamma, *d_output;
    cudaMalloc(&d_input, batch * hidden * sizeof(float));
    cudaMalloc(&d_gamma, hidden * sizeof(float));
    cudaMalloc(&d_output, batch * hidden * sizeof(float));
    
    cudaMemcpy(d_input, h_input, batch * hidden * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, hidden * sizeof(float), cudaMemcpyHostToDevice);
    
    rmsNormKernel<<<batch, BLOCK_SIZE>>>(d_input, d_gamma, d_output, hidden, eps);
    cudaMemcpy(h_output_gpu, d_output, batch * hidden * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify
    float max_diff = 0.0f;
    for (int i = 0; i < batch * hidden; i++) {
        max_diff = fmaxf(max_diff, fabsf(h_output_cpu[i] - h_output_gpu[i]));
    }
    printf("CPU vs GPU max diff: %.2e\n\n", max_diff);
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        rmsNormKernel<<<batch, BLOCK_SIZE>>>(d_input, d_gamma, d_output, hidden, eps);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Performance (%d×%d, 1000 iters): %.2f ms (%.2f us/call)\n", batch, hidden, ms, ms);
    
    printf("\nUsed in: LLaMA, LLaMA 2, Gemma, Mistral, etc.\n");
    
    delete[] h_input; delete[] h_gamma; delete[] h_output_cpu; delete[] h_output_gpu;
    cudaFree(d_input); cudaFree(d_gamma); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
