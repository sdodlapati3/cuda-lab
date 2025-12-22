/**
 * Week 36, Day 1: Multi-Head Attention Structure
 * 
 * MHA allows attending to different positions with different representations.
 * 
 * MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
 * where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

struct MHAConfig {
    int batch;
    int seq_len;
    int num_heads;
    int head_dim;
    int embed_dim;  // = num_heads * head_dim
};

void printMHAShapes(const MHAConfig& cfg) {
    printf("Multi-Head Attention Tensor Shapes:\n\n");
    
    printf("Input:\n");
    printf("  X: [%d, %d, %d]  (batch, seq, embed)\n", 
           cfg.batch, cfg.seq_len, cfg.embed_dim);
    
    printf("\nProjection Weights:\n");
    printf("  W_Q: [%d, %d]  (embed, embed)\n", cfg.embed_dim, cfg.embed_dim);
    printf("  W_K: [%d, %d]  (embed, embed)\n", cfg.embed_dim, cfg.embed_dim);
    printf("  W_V: [%d, %d]  (embed, embed)\n", cfg.embed_dim, cfg.embed_dim);
    printf("  W_O: [%d, %d]  (embed, embed)\n", cfg.embed_dim, cfg.embed_dim);
    
    printf("\nAfter projection & reshape:\n");
    printf("  Q: [%d, %d, %d, %d]  (batch, heads, seq, head_dim)\n",
           cfg.batch, cfg.num_heads, cfg.seq_len, cfg.head_dim);
    printf("  K: [%d, %d, %d, %d]  (batch, heads, seq, head_dim)\n",
           cfg.batch, cfg.num_heads, cfg.seq_len, cfg.head_dim);
    printf("  V: [%d, %d, %d, %d]  (batch, heads, seq, head_dim)\n",
           cfg.batch, cfg.num_heads, cfg.seq_len, cfg.head_dim);
    
    printf("\nAttention computation:\n");
    printf("  QK^T: [%d, %d, %d, %d]  (batch, heads, seq, seq)\n",
           cfg.batch, cfg.num_heads, cfg.seq_len, cfg.seq_len);
    printf("  Attn: [%d, %d, %d, %d]  (batch, heads, seq, seq)\n",
           cfg.batch, cfg.num_heads, cfg.seq_len, cfg.seq_len);
    printf("  Out:  [%d, %d, %d, %d]  (batch, heads, seq, head_dim)\n",
           cfg.batch, cfg.num_heads, cfg.seq_len, cfg.head_dim);
    
    printf("\nFinal:\n");
    printf("  Concat: [%d, %d, %d]  (batch, seq, embed)\n",
           cfg.batch, cfg.seq_len, cfg.embed_dim);
    printf("  Output: [%d, %d, %d]  (batch, seq, embed)\n",
           cfg.batch, cfg.seq_len, cfg.embed_dim);
}

void analyzeMemory(const MHAConfig& cfg) {
    long long qkv_size = 3LL * cfg.batch * cfg.seq_len * cfg.embed_dim;
    long long attn_matrix = (long long)cfg.batch * cfg.num_heads * cfg.seq_len * cfg.seq_len;
    long long output_size = (long long)cfg.batch * cfg.seq_len * cfg.embed_dim;
    
    printf("\nMemory Analysis (float32):\n");
    printf("┌────────────────────┬─────────────────────┬──────────────┐\n");
    printf("│ Tensor             │ Elements            │ Memory (MB)  │\n");
    printf("├────────────────────┼─────────────────────┼──────────────┤\n");
    printf("│ Q, K, V            │ %12lld       │ %10.2f   │\n", 
           qkv_size, qkv_size * 4.0f / (1024*1024));
    printf("│ Attention matrix   │ %12lld       │ %10.2f   │\n",
           attn_matrix, attn_matrix * 4.0f / (1024*1024));
    printf("│ Output             │ %12lld       │ %10.2f   │\n",
           output_size, output_size * 4.0f / (1024*1024));
    printf("├────────────────────┼─────────────────────┼──────────────┤\n");
    printf("│ Total              │                     │ %10.2f   │\n",
           (qkv_size + attn_matrix + output_size) * 4.0f / (1024*1024));
    printf("└────────────────────┴─────────────────────┴──────────────┘\n");
    
    printf("\n⚠️  Attention matrix grows as O(seq²) - this is the problem!\n");
    printf("   seq=1024  → %.1f MB per batch×head\n", 1024*1024*4.0f/(1024*1024));
    printf("   seq=4096  → %.1f MB per batch×head\n", 4096*4096*4.0f/(1024*1024));
    printf("   seq=16384 → %.1f MB per batch×head\n", 16384LL*16384*4.0f/(1024*1024));
}

int main() {
    printf("Week 36 Day 1: Multi-Head Attention Structure\n\n");
    
    printf("Why Multi-Head?\n");
    printf("  • Different heads learn different relationships\n");
    printf("  • Head 1: syntactic patterns\n");
    printf("  • Head 2: semantic similarity\n");
    printf("  • Head 3: positional relationships\n");
    printf("  • etc.\n\n");
    
    // GPT-2 Small config
    MHAConfig gpt2_small = {
        .batch = 8,
        .seq_len = 1024,
        .num_heads = 12,
        .head_dim = 64,
        .embed_dim = 768
    };
    
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("Example: GPT-2 Small\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printMHAShapes(gpt2_small);
    analyzeMemory(gpt2_small);
    
    // LLaMA 7B config
    MHAConfig llama_7b = {
        .batch = 1,
        .seq_len = 4096,
        .num_heads = 32,
        .head_dim = 128,
        .embed_dim = 4096
    };
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Example: LLaMA 7B\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printMHAShapes(llama_7b);
    analyzeMemory(llama_7b);
    
    printf("\nKey Insight:\n");
    printf("  Standard MHA is memory-bound, not compute-bound!\n");
    printf("  We read/write attention matrix multiple times.\n");
    printf("  FlashAttention fixes this by tiling + recomputation.\n");
    
    return 0;
}
