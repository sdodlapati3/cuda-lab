/**
 * Week 34, Day 4: LayerNorm Backward Pass
 * 
 * Gradients: dL/dx, dL/dgamma, dL/dbeta
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

// Backward kernel for dx
__global__ void layerNormBackwardKernel(
    const float* __restrict__ dout,      // Gradient from next layer
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ mean,
    const float* __restrict__ invstd,
    float* __restrict__ dx,
    int hidden
) {
    __shared__ float s_sum1, s_sum2;
    
    int batch_idx = blockIdx.x;
    const float* dy = dout + batch_idx * hidden;
    const float* x = input + batch_idx * hidden;
    float* dx_out = dx + batch_idx * hidden;
    float m = mean[batch_idx];
    float rstd = invstd[batch_idx];
    
    // Compute partial sums for gradient
    float sum1 = 0.0f;  // sum(dy * gamma)
    float sum2 = 0.0f;  // sum(dy * gamma * x_hat)
    
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float x_hat = (x[i] - m) * rstd;
        float dy_gamma = dy[i] * gamma[i];
        sum1 += dy_gamma;
        sum2 += dy_gamma * x_hat;
    }
    
    sum1 = blockReduceSum(sum1);
    sum2 = blockReduceSum(sum2);
    if (threadIdx.x == 0) { s_sum1 = sum1; s_sum2 = sum2; }
    __syncthreads();
    sum1 = s_sum1;
    sum2 = s_sum2;
    
    // Compute dx
    float scale = 1.0f / hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float x_hat = (x[i] - m) * rstd;
        float dy_gamma = dy[i] * gamma[i];
        dx_out[i] = rstd * (dy_gamma - scale * sum1 - scale * x_hat * sum2);
    }
}

// Backward kernel for dgamma and dbeta (reduction across batch)
__global__ void layerNormBackwardParamsKernel(
    const float* __restrict__ dout,
    const float* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ invstd,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int batch, int hidden
) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature_idx >= hidden) return;
    
    float dg = 0.0f, db = 0.0f;
    
    for (int b = 0; b < batch; b++) {
        float dy = dout[b * hidden + feature_idx];
        float x = input[b * hidden + feature_idx];
        float m = mean[b];
        float rstd = invstd[b];
        float x_hat = (x - m) * rstd;
        
        dg += dy * x_hat;
        db += dy;
    }
    
    dgamma[feature_idx] = dg;
    dbeta[feature_idx] = db;
}

int main() {
    printf("Week 34 Day 4: LayerNorm Backward Pass\n\n");
    
    printf("LayerNorm Backward Formulas:\n");
    printf("  x_hat = (x - mean) * invstd\n");
    printf("  \n");
    printf("  dgamma[i] = sum_batch(dout * x_hat)\n");
    printf("  dbeta[i]  = sum_batch(dout)\n");
    printf("  \n");
    printf("  dx = invstd * (dy*gamma - mean(dy*gamma) - x_hat*mean(dy*gamma*x_hat))\n\n");
    
    const int batch = 32, hidden = 512;
    
    // Allocate
    float *h_input = new float[batch * hidden];
    float *h_dout = new float[batch * hidden];
    float *h_gamma = new float[hidden];
    float *h_mean = new float[batch];
    float *h_invstd = new float[batch];
    float *h_dx = new float[batch * hidden];
    float *h_dgamma = new float[hidden];
    float *h_dbeta = new float[hidden];
    
    // Initialize
    for (int i = 0; i < batch * hidden; i++) {
        h_input[i] = (float)(rand() % 100) / 50.0f - 1.0f;
        h_dout[i] = (float)(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < hidden; i++) h_gamma[i] = 1.0f;
    for (int b = 0; b < batch; b++) {
        h_mean[b] = 0.0f;
        h_invstd[b] = 1.0f;  // Simplified
    }
    
    // GPU
    float *d_input, *d_dout, *d_gamma, *d_mean, *d_invstd, *d_dx, *d_dgamma, *d_dbeta;
    cudaMalloc(&d_input, batch * hidden * sizeof(float));
    cudaMalloc(&d_dout, batch * hidden * sizeof(float));
    cudaMalloc(&d_gamma, hidden * sizeof(float));
    cudaMalloc(&d_mean, batch * sizeof(float));
    cudaMalloc(&d_invstd, batch * sizeof(float));
    cudaMalloc(&d_dx, batch * hidden * sizeof(float));
    cudaMalloc(&d_dgamma, hidden * sizeof(float));
    cudaMalloc(&d_dbeta, hidden * sizeof(float));
    
    cudaMemcpy(d_input, h_input, batch * hidden * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dout, h_dout, batch * hidden * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, hidden * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, h_mean, batch * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_invstd, h_invstd, batch * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run backward
    layerNormBackwardKernel<<<batch, BLOCK_SIZE>>>(
        d_dout, d_input, d_gamma, d_mean, d_invstd, d_dx, hidden);
    
    layerNormBackwardParamsKernel<<<(hidden+255)/256, 256>>>(
        d_dout, d_input, d_mean, d_invstd, d_dgamma, d_dbeta, batch, hidden);
    
    cudaMemcpy(h_dx, d_dx, batch * hidden * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dgamma, d_dgamma, hidden * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dbeta, d_dbeta, hidden * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Sample gradients:\n");
    printf("  dx[0..3]: %.4f, %.4f, %.4f, %.4f\n", h_dx[0], h_dx[1], h_dx[2], h_dx[3]);
    printf("  dgamma[0..3]: %.4f, %.4f, %.4f, %.4f\n", h_dgamma[0], h_dgamma[1], h_dgamma[2], h_dgamma[3]);
    printf("  dbeta[0..3]: %.4f, %.4f, %.4f, %.4f\n", h_dbeta[0], h_dbeta[1], h_dbeta[2], h_dbeta[3]);
    
    // Cleanup
    delete[] h_input; delete[] h_dout; delete[] h_gamma;
    delete[] h_mean; delete[] h_invstd; delete[] h_dx;
    delete[] h_dgamma; delete[] h_dbeta;
    cudaFree(d_input); cudaFree(d_dout); cudaFree(d_gamma);
    cudaFree(d_mean); cudaFree(d_invstd); cudaFree(d_dx);
    cudaFree(d_dgamma); cudaFree(d_dbeta);
    
    return 0;
}
