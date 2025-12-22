/**
 * Week 47, Day 2: Tensor Parallelism
 */
#include <cstdio>

int main() {
    printf("Week 47 Day 2: Tensor Parallelism\n\n");
    
    printf("Tensor Parallelism Concept:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ • Split individual layers across GPUs                             ║\n");
    printf("║ • Each GPU holds part of the weight matrix                        ║\n");
    printf("║ • Requires communication within layers                            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Column Parallel Linear:\n");
    printf("  Weight W split by columns: [W1 | W2]\n");
    printf("  GPU0: Y1 = X @ W1\n");
    printf("  GPU1: Y2 = X @ W2\n");
    printf("  Result: Y = [Y1 | Y2] (concatenate)\n\n");
    
    printf("Row Parallel Linear:\n");
    printf("  Weight W split by rows: [W1; W2]\n");
    printf("  Input X split: [X1 | X2]\n");
    printf("  GPU0: Y1 = X1 @ W1\n");
    printf("  GPU1: Y2 = X2 @ W2\n");
    printf("  Result: Y = Y1 + Y2 (AllReduce)\n\n");
    
    printf("When to Use:\n");
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ • Model too large for single GPU memory                            │\n");
    printf("│ • Within NVLink-connected GPUs (low latency needed)                │\n");
    printf("│ • Example: GPT-3 175B uses 8-way tensor parallel                   │\n");
    printf("└─────────────────────────────────────────────────────────────────────┘\n\n");
    
    printf("Megatron-LM Pattern:\n");
    printf("```python\n");
    printf("# Simplified Megatron column parallel\n");
    printf("class ColumnParallelLinear(nn.Module):\n");
    printf("    def __init__(self, in_f, out_f, world_size, rank):\n");
    printf("        self.weight = nn.Parameter(\n");
    printf("            torch.empty(out_f // world_size, in_f))\n");
    printf("    \n");
    printf("    def forward(self, x):\n");
    printf("        local_out = F.linear(x, self.weight)\n");
    printf("        return all_gather(local_out)  # Gather across TP group\n");
    printf("```\n");
    
    return 0;
}
