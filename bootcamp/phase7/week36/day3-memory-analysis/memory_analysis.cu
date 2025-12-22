/**
 * Week 36, Day 3: Memory Traffic Analysis
 * 
 * Understanding IO complexity is key to understanding FlashAttention.
 * Standard attention is memory-bound, not compute-bound.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

struct AttentionIO {
    long long qkt_read;
    long long qkt_write;
    long long softmax_read;
    long long softmax_write;
    long long pv_read;
    long long pv_write;
    long long total_hbm;
};

AttentionIO analyzeStandardAttention(int batch, int heads, int seq, int d) {
    AttentionIO io;
    long long bh = batch * heads;
    
    // QK^T: read Q[bh, seq, d] and K[bh, seq, d], write S[bh, seq, seq]
    io.qkt_read = 2 * bh * seq * d;
    io.qkt_write = bh * seq * seq;
    
    // Softmax: read S, write S (in-place but still counts)
    io.softmax_read = bh * seq * seq;
    io.softmax_write = bh * seq * seq;
    
    // PV: read S[bh, seq, seq] and V[bh, seq, d], write O[bh, seq, d]
    io.pv_read = bh * seq * seq + bh * seq * d;
    io.pv_write = bh * seq * d;
    
    io.total_hbm = (io.qkt_read + io.qkt_write + 
                    io.softmax_read + io.softmax_write +
                    io.pv_read + io.pv_write) * sizeof(float);
    
    return io;
}

AttentionIO analyzeFlashAttention(int batch, int heads, int seq, int d) {
    AttentionIO io;
    long long bh = batch * heads;
    
    // FlashAttention: Q, K, V each read once, O written once
    // No intermediate S matrix to HBM!
    io.qkt_read = bh * seq * d;  // Q
    io.qkt_write = 0;            // No S to HBM
    
    io.softmax_read = bh * seq * d;  // K
    io.softmax_write = 0;
    
    io.pv_read = bh * seq * d;   // V
    io.pv_write = bh * seq * d;  // O
    
    io.total_hbm = (io.qkt_read + io.softmax_read + io.pv_read + io.pv_write) * sizeof(float);
    
    return io;
}

void printComparison(int batch, int heads, int seq, int d) {
    AttentionIO standard = analyzeStandardAttention(batch, heads, seq, d);
    AttentionIO flash = analyzeFlashAttention(batch, heads, seq, d);
    
    printf("┌────────────────────────────────────────────────────────────────┐\n");
    printf("│ Memory Traffic: batch=%d, heads=%d, seq=%d, d=%d            │\n", batch, heads, seq, d);
    printf("├────────────────────┬─────────────────┬─────────────────────────┤\n");
    printf("│ Operation          │ Standard (MB)   │ FlashAttention (MB)     │\n");
    printf("├────────────────────┼─────────────────┼─────────────────────────┤\n");
    printf("│ QK^T read          │ %10.2f      │ %10.2f              │\n",
           standard.qkt_read * 4.0f / (1024*1024), flash.qkt_read * 4.0f / (1024*1024));
    printf("│ QK^T write (S)     │ %10.2f      │ %10.2f (none!)      │\n",
           standard.qkt_write * 4.0f / (1024*1024), flash.qkt_write * 4.0f / (1024*1024));
    printf("│ Softmax R/W        │ %10.2f      │ %10.2f (in SRAM)   │\n",
           (standard.softmax_read + standard.softmax_write) * 4.0f / (1024*1024),
           (flash.softmax_read + flash.softmax_write) * 4.0f / (1024*1024));
    printf("│ PV read            │ %10.2f      │ %10.2f              │\n",
           standard.pv_read * 4.0f / (1024*1024), flash.pv_read * 4.0f / (1024*1024));
    printf("│ PV write           │ %10.2f      │ %10.2f              │\n",
           standard.pv_write * 4.0f / (1024*1024), flash.pv_write * 4.0f / (1024*1024));
    printf("├────────────────────┼─────────────────┼─────────────────────────┤\n");
    printf("│ TOTAL HBM          │ %10.2f      │ %10.2f              │\n",
           standard.total_hbm / (1024.0f*1024), flash.total_hbm / (1024.0f*1024));
    printf("│ Speedup            │       1.0×      │ %10.1f×             │\n",
           (float)standard.total_hbm / flash.total_hbm);
    printf("└────────────────────┴─────────────────┴─────────────────────────┘\n");
}

int main() {
    printf("Week 36 Day 3: Memory Traffic Analysis\n\n");
    
    printf("GPU Memory Hierarchy:\n");
    printf("  ┌─────────────────────────────────────────────┐\n");
    printf("  │ HBM (High Bandwidth Memory)                 │\n");
    printf("  │   Capacity: ~40-80 GB                       │\n");
    printf("  │   Bandwidth: ~1.5-2 TB/s                    │\n");
    printf("  │   Latency: ~400 cycles                      │\n");
    printf("  ├─────────────────────────────────────────────┤\n");
    printf("  │ L2 Cache                                    │\n");
    printf("  │   Capacity: ~6-50 MB                        │\n");
    printf("  │   Bandwidth: ~5-6 TB/s                      │\n");
    printf("  ├─────────────────────────────────────────────┤\n");
    printf("  │ SRAM (Shared Memory / L1)                   │\n");
    printf("  │   Capacity: ~192 KB per SM                  │\n");
    printf("  │   Bandwidth: ~19 TB/s                       │\n");
    printf("  │   Latency: ~30 cycles                       │\n");
    printf("  └─────────────────────────────────────────────┘\n\n");
    
    printf("Standard Attention Memory Flow:\n");
    printf("  1. Read Q, K from HBM → Compute QK^T → Write S to HBM\n");
    printf("  2. Read S from HBM → Softmax → Write S to HBM\n");
    printf("  3. Read S, V from HBM → Compute PV → Write O to HBM\n");
    printf("  ⚠️  S matrix: O(N²) read/write 3 times!\n\n");
    
    printf("FlashAttention Memory Flow:\n");
    printf("  1. Read Q, K, V from HBM (once each)\n");
    printf("  2. Compute attention in SRAM tiles\n");
    printf("  3. Write O to HBM (once)\n");
    printf("  ✓ No O(N²) intermediate storage in HBM!\n\n");
    
    // Compare different sequence lengths
    printf("═══════════════════════════════════════════════════════════════════\n");
    printComparison(8, 12, 512, 64);
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printComparison(8, 12, 2048, 64);
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printComparison(1, 32, 4096, 128);  // LLaMA-like
    
    printf("\nKey Insight:\n");
    printf("  Standard: IO = O(N² + Nd) dominated by N² for long sequences\n");
    printf("  Flash:    IO = O(Nd) - only QKV and output\n");
    printf("  Speedup grows with sequence length!\n");
    
    return 0;
}
