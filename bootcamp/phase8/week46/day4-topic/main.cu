/**
 * Week 46, Day 4: Multi-Node Setup
 */
#include <cstdio>

int main() {
    printf("Week 46 Day 4: Multi-Node Training\n\n");
    
    printf("Network Requirements:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ InfiniBand HDR: ~25 GB/s (preferred)                              ║\n");
    printf("║ Ethernet 100GbE: ~12 GB/s                                         ║\n");
    printf("║ Multiple NICs: NCCL can aggregate bandwidth                       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Launch with torchrun:\n");
    printf("```bash\n");
    printf("# Node 0 (master)\n");
    printf("torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \\\n");
    printf("         --master_addr=node0.example.com --master_port=29500 \\\n");
    printf("         train.py\n");
    printf("\n");
    printf("# Node 1\n");
    printf("torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \\\n");
    printf("         --master_addr=node0.example.com --master_port=29500 \\\n");
    printf("         train.py\n");
    printf("```\n\n");
    
    printf("SLURM Example:\n");
    printf("```bash\n");
    printf("#!/bin/bash\n");
    printf("#SBATCH --nodes=2\n");
    printf("#SBATCH --gpus-per-node=8\n");
    printf("#SBATCH --ntasks-per-node=8\n");
    printf("\n");
    printf("srun torchrun --nproc_per_node=8 train.py\n");
    printf("```\n\n");
    
    printf("Network Tuning:\n");
    printf("```bash\n");
    printf("# Use InfiniBand\n");
    printf("export NCCL_IB_DISABLE=0\n");
    printf("export NCCL_NET_GDR_LEVEL=2  # GPUDirect RDMA\n");
    printf("\n");
    printf("# Socket binding\n");
    printf("export NCCL_SOCKET_IFNAME=ib0  # InfiniBand interface\n");
    printf("```\n");
    
    return 0;
}
