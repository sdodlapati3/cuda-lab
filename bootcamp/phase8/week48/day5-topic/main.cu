/**
 * Week 48, Day 5: Debugging Distributed
 */
#include <cstdio>

int main() {
    printf("Week 48 Day 5: Debugging Distributed Training\n\n");
    
    printf("Common Issues:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ Hang at barrier:   Not all ranks reached it                       ║\n");
    printf("║ NaN loss:          Learning rate too high, gradient explosion     ║\n");
    printf("║ OOM:               Reduce batch size or use FSDP                  ║\n");
    printf("║ Slow training:     Check NCCL_DEBUG for communication issues      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Debug Utilities:\n");
    printf("```python\n");
    printf("# Print from specific rank only\n");
    printf("def print_rank0(msg):\n");
    printf("    if dist.get_rank() == 0:\n");
    printf("        print(msg)\n");
    printf("\n");
    printf("# Detect hang location\n");
    printf("print(f'Rank {rank}: Before forward')\n");
    printf("dist.barrier()\n");
    printf("print(f'Rank {rank}: After forward')\n");
    printf("\n");
    printf("# Check gradients\n");
    printf("for name, param in model.named_parameters():\n");
    printf("    if param.grad is not None:\n");
    printf("        grad_norm = param.grad.norm()\n");
    printf("        if torch.isnan(grad_norm) or grad_norm > 1000:\n");
    printf("            print(f'Bad gradient in {name}: {grad_norm}')\n");
    printf("```\n\n");
    
    printf("Profiling:\n");
    printf("```bash\n");
    printf("# Profile with nsys\n");
    printf("nsys profile -t cuda,nvtx,nccl --output=profile \\\n");
    printf("    torchrun --nproc_per_node=4 train.py\n");
    printf("\n");
    printf("# PyTorch profiler\n");
    printf("with torch.profiler.profile(\n");
    printf("    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],\n");
    printf("    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),\n");
    printf(") as prof:\n");
    printf("    train_loop()\n");
    printf("    prof.step()\n");
    printf("```\n");
    
    return 0;
}
