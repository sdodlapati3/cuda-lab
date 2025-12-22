/**
 * Week 46, Day 3: Async Operations
 */
#include <cstdio>

int main() {
    printf("Week 46 Day 3: Asynchronous Collectives\n\n");
    
    printf("Async Operations:\n");
    printf("```python\n");
    printf("import torch.distributed as dist\n");
    printf("\n");
    printf("# Async all-reduce\n");
    printf("work = dist.all_reduce(tensor, async_op=True)\n");
    printf("\n");
    printf("# Do other work while communication happens\n");
    printf("compute_something_else()\n");
    printf("\n");
    printf("# Wait for completion\n");
    printf("work.wait()\n");
    printf("```\n\n");
    
    printf("Overlap Pattern:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ Timeline without overlap:                                         ║\n");
    printf("║   [Compute]────────────[AllReduce]────────────[Next Compute]      ║\n");
    printf("║                                                                   ║\n");
    printf("║ Timeline with overlap:                                            ║\n");
    printf("║   [Compute]────────────[AllReduce]────────────                    ║\n");
    printf("║                        [Next Compute]─────────                    ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Bucket AllReduce (DDP):\n");
    printf("```python\n");
    printf("# DDP buckets gradients for efficient overlap\n");
    printf("model = DDP(model, \n");
    printf("            bucket_cap_mb=25,  # Bucket size in MB\n");
    printf("            gradient_as_bucket_view=True)  # Memory efficient\n");
    printf("\n");
    printf("# Gradients are reduced as soon as buckets fill\n");
    printf("# Backward computation overlaps with gradient sync!\n");
    printf("```\n");
    
    return 0;
}
