/**
 * Week 37, Day 3: Online Softmax in Attention Context
 * 
 * The key insight: we can update output O incrementally
 * as we process K, V tiles, using running max (m) and sum (l).
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Demonstrates the incremental update formula
void demonstrateRescaling() {
    printf("Online Softmax Rescaling in Attention:\n\n");
    
    printf("Standard softmax over all keys:\n");
    printf("  P = softmax(QK^T) = exp(S - max) / sum(exp(S - max))\n");
    printf("  O = P @ V\n\n");
    
    printf("Problem: Need all S values before computing softmax!\n\n");
    
    printf("Solution: Process tiles incrementally, rescale as we go.\n\n");
    
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ Processing tile j, have accumulated from tiles 0..(j-1):          ║\n");
    printf("║   m_prev = max of scores seen so far                              ║\n");
    printf("║   l_prev = sum of exp(scores - m_prev) seen so far                ║\n");
    printf("║   O_prev = (Σ exp(s_i - m_prev) * v_i) / l_prev   (partial output)║\n");
    printf("║                                                                   ║\n");
    printf("║ After processing tile j:                                          ║\n");
    printf("║   m_curr = max(m_prev, max(S_j))                                  ║\n");
    printf("║   l_curr = l_prev * exp(m_prev - m_curr) + Σexp(S_j - m_curr)     ║\n");
    printf("║   O_curr = O_prev * l_prev * exp(m_prev - m_curr) / l_curr        ║\n");
    printf("║          + (softmax(S_j, m_curr) @ V_j) / l_curr                  ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
}

// CPU demonstration of tiled attention with online softmax
void tiledAttentionCPU(
    const float* Q, const float* K, const float* V, float* O,
    int seq, int d, int Br, int Bc
) {
    // Initialize output and softmax state for each query row
    float* m = new float[seq];  // Running max
    float* l = new float[seq];  // Running sum
    
    for (int i = 0; i < seq; i++) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
        for (int di = 0; di < d; di++) {
            O[i * d + di] = 0.0f;
        }
    }
    
    float scale = 1.0f / sqrtf((float)d);
    
    // Outer loop: K, V tiles
    for (int j = 0; j < seq; j += Bc) {
        int Bc_actual = fmin(Bc, seq - j);
        
        // Inner loop: Q tiles
        for (int i = 0; i < seq; i += Br) {
            int Br_actual = fmin(Br, seq - i);
            
            // For each query in this tile
            for (int qi = i; qi < i + Br_actual; qi++) {
                float m_prev = m[qi];
                float l_prev = l[qi];
                
                // Compute attention scores for this Q row against K tile
                float* S_row = new float[Bc_actual];
                float m_tile = -INFINITY;
                
                for (int kj = 0; kj < Bc_actual; kj++) {
                    int k_idx = j + kj;
                    float score = 0.0f;
                    for (int di = 0; di < d; di++) {
                        score += Q[qi * d + di] * K[k_idx * d + di];
                    }
                    S_row[kj] = score * scale;
                    m_tile = fmaxf(m_tile, S_row[kj]);
                }
                
                // Compute new max
                float m_new = fmaxf(m_prev, m_tile);
                
                // Rescale previous accumulator
                float rescale = expf(m_prev - m_new);
                
                // Compute sum and weighted output for this tile
                float l_tile = 0.0f;
                float* pv_tile = new float[d];
                for (int di = 0; di < d; di++) pv_tile[di] = 0.0f;
                
                for (int kj = 0; kj < Bc_actual; kj++) {
                    int k_idx = j + kj;
                    float p = expf(S_row[kj] - m_new);
                    l_tile += p;
                    for (int di = 0; di < d; di++) {
                        pv_tile[di] += p * V[k_idx * d + di];
                    }
                }
                
                // Update running state
                float l_new = l_prev * rescale + l_tile;
                
                // Update output
                for (int di = 0; di < d; di++) {
                    O[qi * d + di] = (O[qi * d + di] * l_prev * rescale + pv_tile[di]) / l_new;
                }
                
                m[qi] = m_new;
                l[qi] = l_new;
                
                delete[] S_row;
                delete[] pv_tile;
            }
        }
    }
    
    delete[] m;
    delete[] l;
}

// Standard attention for comparison
void standardAttentionCPU(
    const float* Q, const float* K, const float* V, float* O,
    int seq, int d
) {
    float scale = 1.0f / sqrtf((float)d);
    float* S = new float[seq * seq];
    
    // QK^T
    for (int i = 0; i < seq; i++) {
        for (int j = 0; j < seq; j++) {
            float score = 0.0f;
            for (int di = 0; di < d; di++) {
                score += Q[i * d + di] * K[j * d + di];
            }
            S[i * seq + j] = score * scale;
        }
    }
    
    // Softmax
    for (int i = 0; i < seq; i++) {
        float max_val = -INFINITY;
        for (int j = 0; j < seq; j++) max_val = fmaxf(max_val, S[i * seq + j]);
        float sum = 0.0f;
        for (int j = 0; j < seq; j++) {
            S[i * seq + j] = expf(S[i * seq + j] - max_val);
            sum += S[i * seq + j];
        }
        for (int j = 0; j < seq; j++) S[i * seq + j] /= sum;
    }
    
    // PV
    for (int i = 0; i < seq; i++) {
        for (int di = 0; di < d; di++) {
            float out = 0.0f;
            for (int j = 0; j < seq; j++) {
                out += S[i * seq + j] * V[j * d + di];
            }
            O[i * d + di] = out;
        }
    }
    
    delete[] S;
}

int main() {
    printf("Week 37 Day 3: Online Softmax in Attention\n\n");
    
    demonstrateRescaling();
    
    // Test correctness
    const int seq = 64, d = 32;
    const int Br = 16, Bc = 16;
    
    float* Q = new float[seq * d];
    float* K = new float[seq * d];
    float* V = new float[seq * d];
    float* O_std = new float[seq * d];
    float* O_tiled = new float[seq * d];
    
    // Initialize with random data
    for (int i = 0; i < seq * d; i++) {
        Q[i] = (float)(rand() % 100) / 100.0f - 0.5f;
        K[i] = (float)(rand() % 100) / 100.0f - 0.5f;
        V[i] = (float)(rand() % 100) / 100.0f - 0.5f;
    }
    
    standardAttentionCPU(Q, K, V, O_std, seq, d);
    tiledAttentionCPU(Q, K, V, O_tiled, seq, d, Br, Bc);
    
    // Compare
    float max_diff = 0.0f;
    for (int i = 0; i < seq * d; i++) {
        max_diff = fmaxf(max_diff, fabsf(O_std[i] - O_tiled[i]));
    }
    
    printf("Correctness Test:\n");
    printf("  seq=%d, d=%d, Br=%d, Bc=%d\n", seq, d, Br, Bc);
    printf("  Max diff between standard and tiled: %.2e\n", max_diff);
    printf("  Status: %s\n\n", max_diff < 1e-5 ? "✓ PASSED" : "✗ FAILED");
    
    printf("Key Formula (for each Q row):\n");
    printf("  m_new = max(m_prev, max(S_tile))\n");
    printf("  l_new = l_prev * exp(m_prev - m_new) + sum(exp(S_tile - m_new))\n");
    printf("  O_new = (O_prev * l_prev * exp(m_prev - m_new) + P_tile @ V_tile) / l_new\n");
    
    delete[] Q; delete[] K; delete[] V; delete[] O_std; delete[] O_tiled;
    
    return 0;
}
