/**
 * Week 46, Day 2: Process Groups
 */
#include <cstdio>

int main() {
    printf("Week 46 Day 2: Process Groups\n\n");
    
    printf("What are Process Groups?\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ • Subset of ranks that communicate together                       ║\n");
    printf("║ • Enables parallel communication patterns                         ║\n");
    printf("║ • Example: Tensor Parallel uses groups within nodes               ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Creating Process Groups:\n");
    printf("```python\n");
    printf("import torch.distributed as dist\n");
    printf("\n");
    printf("# Default world group (all ranks)\n");
    printf("world_size = dist.get_world_size()\n");
    printf("\n");
    printf("# Create groups for 8 GPUs: [0,1,2,3] and [4,5,6,7]\n");
    printf("group1_ranks = [0, 1, 2, 3]\n");
    printf("group2_ranks = [4, 5, 6, 7]\n");
    printf("\n");
    printf("group1 = dist.new_group(group1_ranks)\n");
    printf("group2 = dist.new_group(group2_ranks)\n");
    printf("\n");
    printf("# Use specific group for collective\n");
    printf("if dist.get_rank() in group1_ranks:\n");
    printf("    dist.all_reduce(tensor, group=group1)\n");
    printf("else:\n");
    printf("    dist.all_reduce(tensor, group=group2)\n");
    printf("```\n\n");
    
    printf("Common Patterns:\n");
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Data Parallel:   All GPUs in one group (default)                   │\n");
    printf("│ Tensor Parallel: Groups within nodes (fast NVLink)                 │\n");
    printf("│ Pipeline Parallel: Groups across pipeline stages                   │\n");
    printf("│ 3D Parallel:     Combine all three!                                │\n");
    printf("└─────────────────────────────────────────────────────────────────────┘\n");
    
    return 0;
}
