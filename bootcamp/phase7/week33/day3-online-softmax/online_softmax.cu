/**
 * Week 33, Day 3: Online Softmax
 * 
 * Key insight: We can update running max and sum in one pass.
 * 
 * When max changes from m to m':
 *   new_sum = old_sum * exp(m - m') + exp(x - m')
 * 
 * Two-pass online algorithm:
 * Pass 1: Compute (max, sum) in one sweep
 * Pass 2: Normalize with final max and sum
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>

// Online softmax state
struct SoftmaxState {
    float max;
    float sum;  // sum of exp(x - max) so far
};

// Combine two states
__host__ __device__ SoftmaxState combineStates(SoftmaxState a, SoftmaxState b) {
    SoftmaxState result;
    result.max = fmaxf(a.max, b.max);
    // Rescale both sums to new max
    result.sum = a.sum * expf(a.max - result.max) + 
                 b.sum * expf(b.max - result.max);
    return result;
}

// CPU online softmax
void onlineSoftmaxCPU(const float* input, float* output, int n) {
    // Pass 1: Online max and sum
    SoftmaxState state = {-FLT_MAX, 0.0f};
    
    for (int i = 0; i < n; i++) {
        float x = input[i];
        float newMax = fmaxf(state.max, x);
        // Rescale existing sum and add new element
        state.sum = state.sum * expf(state.max - newMax) + expf(x - newMax);
        state.max = newMax;
    }
    
    // Pass 2: Normalize
    for (int i = 0; i < n; i++) {
        output[i] = expf(input[i] - state.max) / state.sum;
    }
}

// GPU online softmax kernel
__global__ void onlineSoftmaxKernel(const float* input, float* output, int n) {
    __shared__ float s_max;
    __shared__ float s_sum;
    
    // Pass 1: Each thread computes local (max, sum) online
    SoftmaxState localState = {-FLT_MAX, 0.0f};
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float x = input[i];
        float newMax = fmaxf(localState.max, x);
        localState.sum = localState.sum * expf(localState.max - newMax) + 
                         expf(x - newMax);
        localState.max = newMax;
    }
    
    // Warp reduction of states
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        SoftmaxState other;
        other.max = __shfl_down_sync(0xffffffff, localState.max, offset);
        other.sum = __shfl_down_sync(0xffffffff, localState.sum, offset);
        localState = combineStates(localState, other);
    }
    
    if (threadIdx.x == 0) {
        s_max = localState.max;
        s_sum = localState.sum;
    }
    __syncthreads();
    
    // Pass 2: Normalize
    float maxVal = s_max;
    float sum = s_sum;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        output[i] = expf(input[i] - maxVal) / sum;
    }
}

int main() {
    printf("Week 33 Day 3: Online Softmax\n\n");
    
    printf("Online Algorithm Key Insight:\n");
    printf("  When max changes from m to m':\n");
    printf("  new_sum = old_sum × exp(m - m') + exp(x - m')\n\n");
    
    printf("State combination (for parallel reduction):\n");
    printf("  combined.max = max(a.max, b.max)\n");
    printf("  combined.sum = a.sum × exp(a.max - combined.max)\n");
    printf("              + b.sum × exp(b.max - combined.max)\n\n");
    
    // Test
    const int N = 1024;
    float* h_input = new float[N];
    float* h_output_cpu = new float[N];
    float* h_output_gpu = new float[N];
    
    // Initialize with values that would overflow naive softmax
    for (int i = 0; i < N; i++) {
        h_input[i] = (i % 100) + (i == 500 ? 100.0f : 0.0f);  // One large value
    }
    
    // CPU online softmax
    onlineSoftmaxCPU(h_input, h_output_cpu, N);
    
    // GPU online softmax
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    onlineSoftmaxKernel<<<1, 256>>>(d_input, d_output, N);
    cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify
    float sum_cpu = 0.0f, sum_gpu = 0.0f;
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        sum_cpu += h_output_cpu[i];
        sum_gpu += h_output_gpu[i];
        max_diff = fmaxf(max_diff, fabsf(h_output_cpu[i] - h_output_gpu[i]));
    }
    
    printf("Results (N=%d):\n", N);
    printf("  CPU sum: %.6f\n", sum_cpu);
    printf("  GPU sum: %.6f\n", sum_gpu);
    printf("  Max diff: %.2e\n\n", max_diff);
    
    printf("Pass Comparison:\n");
    printf("  Three-pass: max → exp+sum → normalize\n");
    printf("  Two-pass:   online(max,sum) → normalize\n");
    printf("  Memory traffic reduced by ~33%%\n\n");
    
    printf("This is the foundation for FlashAttention!\n");
    
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
