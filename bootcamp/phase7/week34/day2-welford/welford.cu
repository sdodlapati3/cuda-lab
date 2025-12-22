/**
 * Week 34, Day 2: Welford's Algorithm
 * 
 * Online algorithm for computing mean and variance in one pass.
 * Numerically stable unlike naive sum-of-squares approach.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

struct WelfordState {
    float mean;
    float m2;    // Sum of squared differences from mean
    int count;
};

// Update state with new value
__host__ __device__ WelfordState welfordUpdate(WelfordState state, float x) {
    state.count++;
    float delta = x - state.mean;
    state.mean += delta / state.count;
    float delta2 = x - state.mean;
    state.m2 += delta * delta2;
    return state;
}

// Combine two Welford states (for parallel reduction)
__host__ __device__ WelfordState welfordCombine(WelfordState a, WelfordState b) {
    if (a.count == 0) return b;
    if (b.count == 0) return a;
    
    WelfordState result;
    result.count = a.count + b.count;
    float delta = b.mean - a.mean;
    result.mean = a.mean + delta * b.count / result.count;
    result.m2 = a.m2 + b.m2 + delta * delta * a.count * b.count / result.count;
    return result;
}

__device__ float welfordVariance(WelfordState state) {
    return (state.count > 1) ? state.m2 / state.count : 0.0f;
}

// GPU kernel using Welford's algorithm
__global__ void welfordLayerNormKernel(const float* input, const float* gamma,
                                        const float* beta, float* output,
                                        int hidden, float eps) {
    __shared__ float s_mean;
    __shared__ float s_var;
    
    int batch_idx = blockIdx.x;
    const float* x = input + batch_idx * hidden;
    float* y = output + batch_idx * hidden;
    
    // Each thread computes local Welford state
    WelfordState localState = {0.0f, 0.0f, 0};
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        localState = welfordUpdate(localState, x[i]);
    }
    
    // Warp reduction
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        WelfordState other;
        other.mean = __shfl_down_sync(0xffffffff, localState.mean, offset);
        other.m2 = __shfl_down_sync(0xffffffff, localState.m2, offset);
        other.count = __shfl_down_sync(0xffffffff, localState.count, offset);
        localState = welfordCombine(localState, other);
    }
    
    if (threadIdx.x == 0) {
        s_mean = localState.mean;
        s_var = welfordVariance(localState);
    }
    __syncthreads();
    
    float mean = s_mean;
    float invStd = rsqrtf(s_var + eps);
    
    // Normalize
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        y[i] = gamma[i] * (x[i] - mean) * invStd + beta[i];
    }
}

int main() {
    printf("Week 34 Day 2: Welford's Algorithm\n\n");
    
    printf("Welford's Online Algorithm:\n");
    printf("  For each new value x:\n");
    printf("    count++\n");
    printf("    delta = x - mean\n");
    printf("    mean += delta / count\n");
    printf("    delta2 = x - mean  (note: updated mean)\n");
    printf("    m2 += delta * delta2\n");
    printf("  variance = m2 / count\n\n");
    
    printf("Why Welford?\n");
    printf("  Naive: var = E[x²] - E[x]²\n");
    printf("  Problem: Catastrophic cancellation for large values\n");
    printf("  Welford: Numerically stable, single pass\n\n");
    
    // Test combining states
    printf("Parallel Combination:\n");
    WelfordState a = {10.0f, 20.0f, 5};   // mean=10, m2=20, n=5
    WelfordState b = {20.0f, 30.0f, 5};   // mean=20, m2=30, n=5
    WelfordState combined = welfordCombine(a, b);
    printf("  State A: mean=%.1f, m2=%.1f, n=%d\n", a.mean, a.m2, a.count);
    printf("  State B: mean=%.1f, m2=%.1f, n=%d\n", b.mean, b.m2, b.count);
    printf("  Combined: mean=%.1f, m2=%.1f, n=%d\n\n", 
           combined.mean, combined.m2, combined.count);
    
    // GPU test
    const int batch = 4, hidden = 256;
    float *h_input = new float[batch * hidden];
    float *h_gamma = new float[hidden];
    float *h_beta = new float[hidden];
    float *h_output = new float[batch * hidden];
    
    for (int i = 0; i < batch * hidden; i++) h_input[i] = (float)(i % 100) + 1000.0f;
    for (int i = 0; i < hidden; i++) { h_gamma[i] = 1.0f; h_beta[i] = 0.0f; }
    
    float *d_input, *d_gamma, *d_beta, *d_output;
    cudaMalloc(&d_input, batch * hidden * sizeof(float));
    cudaMalloc(&d_gamma, hidden * sizeof(float));
    cudaMalloc(&d_beta, hidden * sizeof(float));
    cudaMalloc(&d_output, batch * hidden * sizeof(float));
    
    cudaMemcpy(d_input, h_input, batch * hidden * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, hidden * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, hidden * sizeof(float), cudaMemcpyHostToDevice);
    
    welfordLayerNormKernel<<<batch, 32>>>(d_input, d_gamma, d_beta, d_output, hidden, 1e-5f);
    cudaMemcpy(h_output, d_output, batch * hidden * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify
    float mean = 0, var = 0;
    for (int i = 0; i < hidden; i++) mean += h_output[i];
    mean /= hidden;
    for (int i = 0; i < hidden; i++) var += (h_output[i] - mean) * (h_output[i] - mean);
    var /= hidden;
    
    printf("GPU Result (batch 0):\n");
    printf("  Output mean: %.6f (should be ~0)\n", mean);
    printf("  Output var:  %.6f (should be ~1)\n", var);
    
    delete[] h_input; delete[] h_gamma; delete[] h_beta; delete[] h_output;
    cudaFree(d_input); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_output);
    return 0;
}
