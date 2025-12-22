/**
 * Week 45, Day 4: PyTorch Distributed
 */
#include <cstdio>

int main() {
    printf("Week 45 Day 4: PyTorch Distributed with NCCL\n\n");
    
    printf("Process Group Setup:\n");
    printf("```python\n");
    printf("import torch.distributed as dist\n");
    printf("import os\n");
    printf("\n");
    printf("def setup(rank, world_size):\n");
    printf("    os.environ['MASTER_ADDR'] = 'localhost'\n");
    printf("    os.environ['MASTER_PORT'] = '12355'\n");
    printf("    dist.init_process_group('nccl', rank=rank, world_size=world_size)\n");
    printf("    torch.cuda.set_device(rank)\n");
    printf("\n");
    printf("def cleanup():\n");
    printf("    dist.destroy_process_group()\n");
    printf("```\n\n");
    
    printf("DistributedDataParallel (DDP):\n");
    printf("```python\n");
    printf("from torch.nn.parallel import DistributedDataParallel as DDP\n");
    printf("\n");
    printf("def train(rank, world_size):\n");
    printf("    setup(rank, world_size)\n");
    printf("    \n");
    printf("    model = MyModel().to(rank)\n");
    printf("    ddp_model = DDP(model, device_ids=[rank])\n");
    printf("    \n");
    printf("    optimizer = torch.optim.Adam(ddp_model.parameters())\n");
    printf("    \n");
    printf("    for batch in dataloader:\n");
    printf("        optimizer.zero_grad()\n");
    printf("        loss = ddp_model(batch).sum()\n");
    printf("        loss.backward()  # Gradients auto-synced via NCCL!\n");
    printf("        optimizer.step()\n");
    printf("    \n");
    printf("    cleanup()\n");
    printf("\n");
    printf("# Launch with torchrun\n");
    printf("# torchrun --nproc_per_node=4 train.py\n");
    printf("```\n\n");
    
    printf("DDP Under the Hood:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ 1. Model replicated to each GPU                                   ║\n");
    printf("║ 2. Each GPU processes different data batch                        ║\n");
    printf("║ 3. Gradients computed locally                                     ║\n");
    printf("║ 4. NCCL AllReduce averages gradients across GPUs                  ║\n");
    printf("║ 5. All GPUs apply same update → stay synchronized                 ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
