/**
 * Week 38, Day 5: Benchmarking Guide
 */
#include <cstdio>

int main() {
    printf("Week 38 Day 5: Benchmarking FlashAttention\n\n");
    
    printf("Benchmarking Considerations:\n");
    printf("  • Warm up GPU before timing\n");
    printf("  • Use CUDA events for accurate timing\n");
    printf("  • Test multiple sequence lengths\n");
    printf("  • Compare memory usage (torch.cuda.max_memory_allocated)\n\n");
    
    printf("Python Benchmark Template:\n");
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ import torch                                                    │\n");
    printf("│ import time                                                     │\n");
    printf("│                                                                 │\n");
    printf("│ def benchmark_attention(batch, heads, seq, dim, causal=True):   │\n");
    printf("│     q = torch.randn(batch, heads, seq, dim, device='cuda')      │\n");
    printf("│     k = torch.randn(batch, heads, seq, dim, device='cuda')      │\n");
    printf("│     v = torch.randn(batch, heads, seq, dim, device='cuda')      │\n");
    printf("│                                                                 │\n");
    printf("│     # Warmup                                                    │\n");
    printf("│     for _ in range(10):                                         │\n");
    printf("│         _ = F.scaled_dot_product_attention(q, k, v)             │\n");
    printf("│     torch.cuda.synchronize()                                    │\n");
    printf("│                                                                 │\n");
    printf("│     # Benchmark                                                 │\n");
    printf("│     torch.cuda.reset_peak_memory_stats()                        │\n");
    printf("│     start = torch.cuda.Event(enable_timing=True)                │\n");
    printf("│     end = torch.cuda.Event(enable_timing=True)                  │\n");
    printf("│                                                                 │\n");
    printf("│     start.record()                                              │\n");
    printf("│     for _ in range(100):                                        │\n");
    printf("│         _ = F.scaled_dot_product_attention(q, k, v)             │\n");
    printf("│     end.record()                                                │\n");
    printf("│     torch.cuda.synchronize()                                    │\n");
    printf("│                                                                 │\n");
    printf("│     time_ms = start.elapsed_time(end) / 100                     │\n");
    printf("│     memory_mb = torch.cuda.max_memory_allocated() / 1e6         │\n");
    printf("│     return time_ms, memory_mb                                   │\n");
    printf("└─────────────────────────────────────────────────────────────────┘\n");
    
    return 0;
}
