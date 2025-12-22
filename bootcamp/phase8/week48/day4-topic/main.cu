/**
 * Week 48, Day 4: Production Training
 */
#include <cstdio>

int main() {
    printf("Week 48 Day 4: Production-Scale Training\n\n");
    
    printf("Complete Training Script Structure:\n");
    printf("```python\n");
    printf("def main():\n");
    printf("    # 1. Initialize distributed\n");
    printf("    dist.init_process_group('nccl')\n");
    printf("    rank = dist.get_rank()\n");
    printf("    world_size = dist.get_world_size()\n");
    printf("    torch.cuda.set_device(rank)\n");
    printf("    \n");
    printf("    # 2. Create model\n");
    printf("    model = MyModel().cuda()\n");
    printf("    model = DDP(model, device_ids=[rank])\n");
    printf("    \n");
    printf("    # 3. Create distributed sampler\n");
    printf("    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)\n");
    printf("    dataloader = DataLoader(dataset, sampler=sampler, batch_size=bs)\n");
    printf("    \n");
    printf("    # 4. Training loop\n");
    printf("    for epoch in range(epochs):\n");
    printf("        sampler.set_epoch(epoch)  # Important for shuffling!\n");
    printf("        train_epoch(model, dataloader, optimizer)\n");
    printf("        \n");
    printf("        # 5. Checkpoint periodically\n");
    printf("        if rank == 0 and epoch %% save_interval == 0:\n");
    printf("            save_checkpoint(model, optimizer, epoch)\n");
    printf("    \n");
    printf("    dist.destroy_process_group()\n");
    printf("\n");
    printf("if __name__ == '__main__':\n");
    printf("    main()\n");
    printf("```\n\n");
    
    printf("Launch:\n");
    printf("```bash\n");
    printf("# Single node, 8 GPUs\n");
    printf("torchrun --nproc_per_node=8 train.py\n");
    printf("\n");
    printf("# Multi-node\n");
    printf("torchrun --nnodes=2 --nproc_per_node=8 \\\n");
    printf("    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \\\n");
    printf("    train.py\n");
    printf("```\n");
    
    return 0;
}
