/**
 * Week 45, Day 6: Week Summary
 */
#include <cstdio>

int main() {
    printf("Week 45 Summary: NCCL Fundamentals\n");
    printf("══════════════════════════════════════════════════════════════════════\n\n");
    
    printf("Day 1: NCCL Introduction\n");
    printf("  • Collective operations: AllReduce, AllGather, ReduceScatter\n");
    printf("  • Foundation for distributed training\n\n");
    
    printf("Day 2: Ring and Tree Algorithms\n");
    printf("  • Ring: bandwidth-optimal for large messages\n");
    printf("  • Tree: latency-optimal for small messages\n\n");
    
    printf("Day 3: NCCL C API\n");
    printf("  • ncclCommInitRank, ncclAllReduce, etc.\n");
    printf("  • ncclGroupStart/End for batching\n\n");
    
    printf("Day 4: PyTorch Distributed\n");
    printf("  • dist.init_process_group('nccl')\n");
    printf("  • DDP auto-syncs gradients\n\n");
    
    printf("Day 5: Performance Tuning\n");
    printf("  • NCCL_DEBUG, NCCL_ALGO environment variables\n");
    printf("  • nccl-tests for benchmarking\n\n");
    
    printf("Key Takeaways:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ • NCCL handles topology automatically                             ║\n");
    printf("║ • Ring AllReduce scales to many GPUs efficiently                  ║\n");
    printf("║ • DDP is the easiest path to multi-GPU training                   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
