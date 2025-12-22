/**
 * Week 37, Day 4: Simplified FlashAttention Forward
 * 
 * Single kernel that computes attention without materializing S matrix.
 * This is a teaching implementation, not production-optimized.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define Br 32  // Query tile size
#define Bc 32  // Key/Value tile size

__global__ void flashAttentionForward(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L,  // Store logsumexp for backward
    int seq, int d, float scale
) {
    // Each block handles Br query rows
    int q_start = blockIdx.x * Br;
    
    extern __shared__ float smem[];
    float* s_Q = smem;                      // [Br, d]
    float* s_K = s_Q + Br * d;              // [Bc, d]
    float* s_V = s_K + Bc * d;              // [Bc, d]
    float* s_S = s_V + Bc * d;              // [Br, Bc]
    float* s_O = s_S + Br * Bc;             // [Br, d]
    float* s_m = s_O + Br * d;              // [Br]
    float* s_l = s_m + Br;                  // [Br]
    
    // Thread indices
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Initialize output and softmax state
    for (int i = tid; i < Br * d; i += num_threads) {
        s_O[i] = 0.0f;
    }
    for (int i = tid; i < Br; i += num_threads) {
        s_m[i] = -INFINITY;
        s_l[i] = 0.0f;
    }
    __syncthreads();
    
    // Load Q tile (persistent in SRAM)
    for (int i = tid; i < Br * d; i += num_threads) {
        int row = i / d;
        int col = i % d;
        int global_row = q_start + row;
        if (global_row < seq) {
            s_Q[i] = Q[global_row * d + col];
        } else {
            s_Q[i] = 0.0f;
        }
    }
    __syncthreads();
    
    // Loop over K, V tiles
    for (int k_start = 0; k_start < seq; k_start += Bc) {
        // Load K, V tiles
        for (int i = tid; i < Bc * d; i += num_threads) {
            int row = i / d;
            int col = i % d;
            int global_row = k_start + row;
            if (global_row < seq) {
                s_K[i] = K[global_row * d + col];
                s_V[i] = V[global_row * d + col];
            } else {
                s_K[i] = 0.0f;
                s_V[i] = 0.0f;
            }
        }
        __syncthreads();
        
        // Compute S = Q @ K^T for this tile
        for (int i = tid; i < Br * Bc; i += num_threads) {
            int qi = i / Bc;
            int kj = i % Bc;
            
            float score = 0.0f;
            for (int di = 0; di < d; di++) {
                score += s_Q[qi * d + di] * s_K[kj * d + di];
            }
            s_S[i] = score * scale;
        }
        __syncthreads();
        
        // Update online softmax and output for each query row
        // (Simplified: each thread handles some rows)
        for (int qi = tid; qi < Br; qi += num_threads) {
            if (q_start + qi >= seq) continue;
            
            float m_prev = s_m[qi];
            float l_prev = s_l[qi];
            
            // Find max in this tile
            float m_tile = -INFINITY;
            for (int kj = 0; kj < Bc && k_start + kj < seq; kj++) {
                m_tile = fmaxf(m_tile, s_S[qi * Bc + kj]);
            }
            
            float m_new = fmaxf(m_prev, m_tile);
            
            // Compute exp and sum for this tile
            float l_tile = 0.0f;
            for (int kj = 0; kj < Bc && k_start + kj < seq; kj++) {
                s_S[qi * Bc + kj] = expf(s_S[qi * Bc + kj] - m_new);
                l_tile += s_S[qi * Bc + kj];
            }
            
            // Rescale previous accumulator
            float rescale = expf(m_prev - m_new);
            float l_new = l_prev * rescale + l_tile;
            
            // Update output: O = (O * l_prev * rescale + P @ V) / l_new
            for (int di = 0; di < d; di++) {
                float pv = 0.0f;
                for (int kj = 0; kj < Bc && k_start + kj < seq; kj++) {
                    pv += s_S[qi * Bc + kj] * s_V[kj * d + di];
                }
                s_O[qi * d + di] = (s_O[qi * d + di] * l_prev * rescale + pv) / l_new;
            }
            
            s_m[qi] = m_new;
            s_l[qi] = l_new;
        }
        __syncthreads();
    }
    
    // Write output
    for (int i = tid; i < Br * d; i += num_threads) {
        int row = i / d;
        int col = i % d;
        int global_row = q_start + row;
        if (global_row < seq) {
            O[global_row * d + col] = s_O[i];
        }
    }
    
    // Store logsumexp for backward pass
    for (int i = tid; i < Br; i += num_threads) {
        int global_row = q_start + i;
        if (global_row < seq) {
            L[global_row] = s_m[i] + logf(s_l[i]);
        }
    }
}

int main() {
    printf("Week 37 Day 4: FlashAttention Forward\n\n");
    
    const int seq = 256, d = 64;
    const float scale = 1.0f / sqrtf((float)d);
    
    printf("Configuration: seq=%d, d=%d, Br=%d, Bc=%d\n\n", seq, d, Br, Bc);
    
    // Calculate shared memory
    int smem_size = (Br * d + 2 * Bc * d + Br * Bc + Br * d + 2 * Br) * sizeof(float);
    printf("Shared memory per block: %.1f KB\n\n", smem_size / 1024.0f);
    
    // Allocate
    float *h_Q = new float[seq * d];
    float *h_K = new float[seq * d];
    float *h_V = new float[seq * d];
    float *h_O = new float[seq * d];
    
    for (int i = 0; i < seq * d; i++) {
        h_Q[i] = (float)(rand() % 100) / 100.0f - 0.5f;
        h_K[i] = (float)(rand() % 100) / 100.0f - 0.5f;
        h_V[i] = (float)(rand() % 100) / 100.0f - 0.5f;
    }
    
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, seq * d * sizeof(float));
    cudaMalloc(&d_K, seq * d * sizeof(float));
    cudaMalloc(&d_V, seq * d * sizeof(float));
    cudaMalloc(&d_O, seq * d * sizeof(float));
    cudaMalloc(&d_L, seq * sizeof(float));
    
    cudaMemcpy(d_Q, h_Q, seq * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, seq * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, seq * d * sizeof(float), cudaMemcpyHostToDevice);
    
    int num_blocks = (seq + Br - 1) / Br;
    
    printf("Launching %d blocks with %d threads each\n", num_blocks, 256);
    
    flashAttentionForward<<<num_blocks, 256, smem_size>>>(
        d_Q, d_K, d_V, d_O, d_L, seq, d, scale
    );
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_O, d_O, seq * d * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print sample output
    printf("\nOutput sample (first row): [");
    for (int i = 0; i < 8; i++) printf("%.3f ", h_O[i]);
    printf("...]\n\n");
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        flashAttentionForward<<<num_blocks, 256, smem_size>>>(
            d_Q, d_K, d_V, d_O, d_L, seq, d, scale
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Performance: %.2f us/call\n", ms);
    
    printf("\nNote: This is a teaching implementation.\n");
    printf("Production FlashAttention uses:\n");
    printf("  • Better thread mapping\n");
    printf("  • Vectorized loads (float4)\n");
    printf("  • Tensor cores for matmul\n");
    printf("  • More aggressive tiling\n");
    
    delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_O;
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
