/**
 * Week 46, Day 6: Week Summary
 */
#include <cstdio>

int main() {
    printf("Week 46 Summary: NCCL Advanced\n");
    printf("══════════════════════════════════════════════════════════════════════\n\n");
    
    printf("Day 1: Debugging\n");
    printf("  • NCCL_DEBUG=INFO/TRACE/WARN\n");
    printf("  • Common issues: hangs, timeouts, OOM\n\n");
    
    printf("Day 2: Process Groups\n");
    printf("  • dist.new_group() for subsets\n");
    printf("  • Tensor/Pipeline/Data parallel groups\n\n");
    
    printf("Day 3: Async Operations\n");
    printf("  • async_op=True for overlap\n");
    printf("  • work.wait() to synchronize\n\n");
    
    printf("Day 4: Multi-Node Setup\n");
    printf("  • torchrun with --nnodes\n");
    printf("  • InfiniBand, GPUDirect RDMA\n\n");
    
    printf("Day 5: Optimization\n");
    printf("  • Bucket tuning, static_graph\n");
    printf("  • PowerSGD compression\n\n");
    
    printf("Key Takeaways:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ • NCCL_DEBUG is your friend for troubleshooting                   ║\n");
    printf("║ • Process groups enable advanced parallelism                      ║\n");
    printf("║ • Overlap communication with compute for best throughput          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
