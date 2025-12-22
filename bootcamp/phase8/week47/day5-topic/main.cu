/**
 * Week 47, Day 5: 3D Parallelism
 */
#include <cstdio>

int main() {
    printf("Week 47 Day 5: 3D Parallelism\n\n");
    
    printf("3D Parallelism = DP + TP + PP:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ Example: 64 GPUs for 175B model                                   ║\n");
    printf("║                                                                   ║\n");
    printf("║ • Tensor Parallel: 8 GPUs per layer (within node, NVLink)         ║\n");
    printf("║ • Pipeline Parallel: 8 stages (across nodes)                      ║\n");
    printf("║ • Data Parallel: 1 replica (64/8/8 = 1)                           ║\n");
    printf("║                                                                   ║\n");
    printf("║ Or with more GPUs:                                                ║\n");
    printf("║ • TP=8, PP=8, DP=4 → 256 GPUs                                     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Visual (TP=2, PP=2, DP=2 = 8 GPUs):\n");
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Data Replica 0:                                                    │\n");
    printf("│   Stage 0: [GPU0, GPU1] (tensor parallel)                          │\n");
    printf("│   Stage 1: [GPU2, GPU3] (tensor parallel)                          │\n");
    printf("│                                                                    │\n");
    printf("│ Data Replica 1:                                                    │\n");
    printf("│   Stage 0: [GPU4, GPU5] (tensor parallel)                          │\n");
    printf("│   Stage 1: [GPU6, GPU7] (tensor parallel)                          │\n");
    printf("└─────────────────────────────────────────────────────────────────────┘\n\n");
    
    printf("Megatron-DeepSpeed:\n");
    printf("```bash\n");
    printf("deepspeed --num_gpus 64 train.py \\\n");
    printf("    --tensor-model-parallel-size 8 \\\n");
    printf("    --pipeline-model-parallel-size 8 \\\n");
    printf("    --deepspeed_config ds_config.json\n");
    printf("```\n");
    
    return 0;
}
