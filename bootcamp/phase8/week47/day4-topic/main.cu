/**
 * Week 47, Day 4: FSDP (Fully Sharded Data Parallel)
 */
#include <cstdio>

int main() {
    printf("Week 47 Day 4: FSDP\n\n");
    
    printf("FSDP Concept:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ • Shard model parameters across GPUs                              ║\n");
    printf("║ • All-gather parameters just before forward                       ║\n");
    printf("║ • Reduce-scatter gradients after backward                         ║\n");
    printf("║ • Memory efficient: each GPU holds 1/N of model                   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Memory Comparison (12B model, 4 GPUs):\n");
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ DDP:  Each GPU: 12B params + 12B grads + 24B optimizer = 48B      │\n");
    printf("│ FSDP: Each GPU:  3B params +  3B grads +  6B optimizer = 12B      │\n");
    printf("└─────────────────────────────────────────────────────────────────────┘\n\n");
    
    printf("PyTorch FSDP:\n");
    printf("```python\n");
    printf("from torch.distributed.fsdp import (\n");
    printf("    FullyShardedDataParallel as FSDP,\n");
    printf("    ShardingStrategy,\n");
    printf(")\n");
    printf("\n");
    printf("model = FSDP(\n");
    printf("    model,\n");
    printf("    sharding_strategy=ShardingStrategy.FULL_SHARD,\n");
    printf("    cpu_offload=None,  # Can offload to CPU for huge models\n");
    printf("    auto_wrap_policy=transformer_auto_wrap_policy,\n");
    printf(")\n");
    printf("\n");
    printf("# Training loop is same as DDP!\n");
    printf("for batch in dataloader:\n");
    printf("    loss = model(batch).sum()\n");
    printf("    loss.backward()\n");
    printf("    optimizer.step()\n");
    printf("```\n\n");
    
    printf("Sharding Strategies:\n");
    printf("  FULL_SHARD:     Shard everything (most memory efficient)\n");
    printf("  SHARD_GRAD_OP:  Shard only gradients and optimizer\n");
    printf("  NO_SHARD:       Like DDP (baseline)\n");
    
    return 0;
}
