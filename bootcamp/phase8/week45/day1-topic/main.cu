/**
 * Week 45, Day 1: NCCL Introduction
 */
#include <cstdio>

int main() {
    printf("Week 45 Day 1: NCCL Introduction\n\n");
    
    printf("What is NCCL?\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ NVIDIA Collective Communication Library                           ║\n");
    printf("║ • Optimized multi-GPU and multi-node communication                ║\n");
    printf("║ • Uses NVLink, PCIe, InfiniBand automatically                     ║\n");
    printf("║ • Foundation for PyTorch DDP, Horovod, etc.                       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Core Collective Operations:\n");
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ AllReduce     - Sum tensors across all GPUs, result on all         │\n");
    printf("│ Broadcast     - One GPU sends to all others                        │\n");
    printf("│ AllGather     - Each GPU contributes, all get full result          │\n");
    printf("│ ReduceScatter - Reduce + distribute different parts to GPUs        │\n");
    printf("│ Send/Recv     - Point-to-point communication                       │\n");
    printf("└─────────────────────────────────────────────────────────────────────┘\n\n");
    
    printf("AllReduce Visual (4 GPUs):\n");
    printf("  Before:  GPU0: [1,2]  GPU1: [3,4]  GPU2: [5,6]  GPU3: [7,8]\n");
    printf("  After:   GPU0: [16,20] GPU1: [16,20] GPU2: [16,20] GPU3: [16,20]\n");
    printf("  (1+3+5+7=16, 2+4+6+8=20)\n\n");
    
    printf("PyTorch Usage:\n");
    printf("```python\n");
    printf("import torch.distributed as dist\n");
    printf("\n");
    printf("# Initialize\n");
    printf("dist.init_process_group(backend='nccl')\n");
    printf("\n");
    printf("# AllReduce\n");
    printf("tensor = torch.randn(1024, device='cuda')\n");
    printf("dist.all_reduce(tensor, op=dist.ReduceOp.SUM)\n");
    printf("\n");
    printf("# AllGather\n");
    printf("output_list = [torch.empty_like(tensor) for _ in range(world_size)]\n");
    printf("dist.all_gather(output_list, tensor)\n");
    printf("```\n");
    
    return 0;
}
