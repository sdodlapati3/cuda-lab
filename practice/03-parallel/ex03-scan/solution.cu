#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define BLOCK_SIZE 256

// SOLUTION: Hillis-Steele inclusive scan
__global__ void scan_hillis_steele(float* out, const float* in, int n) {
    __shared__ float temp[2 * BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int pout = 0, pin = 1;
    temp[tid] = (idx < n) ? in[idx] : 0;
    __syncthreads();
    
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        pout = 1 - pout;
        pin = 1 - pin;
        
        if (tid >= offset) {
            temp[pout * BLOCK_SIZE + tid] = temp[pin * BLOCK_SIZE + tid] + temp[pin * BLOCK_SIZE + tid - offset];
        } else {
            temp[pout * BLOCK_SIZE + tid] = temp[pin * BLOCK_SIZE + tid];
        }
        __syncthreads();
    }
    
    if (idx < n) {
        out[idx] = temp[pout * BLOCK_SIZE + tid];
    }
}

// SOLUTION: Blelloch exclusive scan
__global__ void scan_blelloch(float* out, const float* in, int n) {
    __shared__ float temp[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    temp[tid] = (idx < n) ? in[idx] : 0;
    __syncthreads();
    
    // Up-sweep (reduce phase)
    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }
    
    // Set last element to 0 for exclusive scan
    if (tid == 0) {
        temp[BLOCK_SIZE - 1] = 0;
    }
    __syncthreads();
    
    // Down-sweep
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE) {
            float t = temp[index - stride];
            temp[index - stride] = temp[index];
            temp[index] += t;
        }
        __syncthreads();
    }
    
    if (idx < n) {
        out[idx] = temp[tid];
    }
}

void cpu_scan_inclusive(float* out, const float* in, int n) {
    out[0] = in[0];
    for (int i = 1; i < n; i++) {
        out[i] = out[i-1] + in[i];
    }
}

void cpu_scan_exclusive(float* out, const float* in, int n) {
    out[0] = 0;
    for (int i = 1; i < n; i++) {
        out[i] = out[i-1] + in[i-1];
    }
}

int main() {
    const int N = BLOCK_SIZE;
    
    printf("Prefix Sum (Scan) - SOLUTION\n");
    printf("============================\n");
    
    float *h_in, *h_out, *h_ref_inc, *h_ref_exc;
    float *d_in, *d_out;
    
    h_in = (float*)malloc(N * sizeof(float));
    h_out = (float*)malloc(N * sizeof(float));
    h_ref_inc = (float*)malloc(N * sizeof(float));
    h_ref_exc = (float*)malloc(N * sizeof(float));
    
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;
    cpu_scan_inclusive(h_ref_inc, h_in, N);
    cpu_scan_exclusive(h_ref_exc, h_in, N);
    
    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Test Hillis-Steele
    scan_hillis_steele<<<1, BLOCK_SIZE>>>(d_out, d_in, N);
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool hs_correct = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_ref_inc[i]) { hs_correct = false; break; }
    }
    printf("Hillis-Steele (inclusive): %s\n", hs_correct ? "PASSED" : "FAILED");
    printf("  Last element: %.0f (expected %.0f)\n", h_out[N-1], h_ref_inc[N-1]);
    
    // Test Blelloch
    scan_blelloch<<<1, BLOCK_SIZE>>>(d_out, d_in, N);
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool bl_correct = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_ref_exc[i]) { bl_correct = false; break; }
    }
    printf("Blelloch (exclusive): %s\n", bl_correct ? "PASSED" : "FAILED");
    printf("  Last element: %.0f (expected %.0f)\n", h_out[N-1], h_ref_exc[N-1]);
    
    free(h_in); free(h_out); free(h_ref_inc); free(h_ref_exc);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    
    printf("\n✓ Solution complete!\n");
    return 0;
}
