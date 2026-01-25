#!/usr/bin/env python3
"""
collective_ops.py - Interactive demonstration of NCCL collective operations

Demonstrates:
- All-reduce, All-gather, Reduce-scatter
- Broadcast, Scatter, Gather
- Send/Recv (point-to-point)
- Process groups and sub-groups

Usage:
    torchrun --nproc_per_node=4 collective_ops.py
    
    # With NCCL debugging
    NCCL_DEBUG=INFO torchrun --nproc_per_node=4 collective_ops.py

Author: CUDA Lab
"""

import os
import time
import argparse
from typing import List, Optional

import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def print_tensor(name: str, tensor: torch.Tensor, rank: int, show_all: bool = False):
    """Print tensor info across ranks."""
    dist.barrier()
    
    if show_all or rank == 0:
        for r in range(dist.get_world_size()):
            if rank == r:
                if tensor.numel() <= 16:
                    print(f"  [Rank {rank}] {name}: {tensor.tolist()}")
                else:
                    print(f"  [Rank {rank}] {name}: shape={list(tensor.shape)}, "
                          f"sum={tensor.sum().item():.2f}")
            dist.barrier()
    else:
        for r in range(dist.get_world_size()):
            dist.barrier()


# ============================================================================
# ALL-REDUCE: Sum/Average across all GPUs, result on all GPUs
# ============================================================================

def demo_all_reduce(rank: int, world_size: int):
    """
    All-Reduce: Combine data from all GPUs and distribute result to all.
    
    Used in: DDP gradient synchronization
    
    Example: Sum gradients
        Before: GPU0=[1,2], GPU1=[3,4], GPU2=[5,6], GPU3=[7,8]
        After:  GPU0=[16,20], GPU1=[16,20], GPU2=[16,20], GPU3=[16,20]
    """
    if rank == 0:
        print("\n" + "=" * 60)
        print("ALL-REDUCE OPERATION")
        print("=" * 60)
        print("Combines values from all ranks, result available on all ranks")
        print("Used by: DDP for gradient synchronization")
    
    # Each rank has unique data
    tensor = torch.tensor([rank * 2 + 1, rank * 2 + 2], dtype=torch.float32).cuda()
    
    if rank == 0:
        print("\nBefore all_reduce (SUM):")
    print_tensor("data", tensor, rank, show_all=True)
    
    # All-reduce with SUM
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        print("\nAfter all_reduce (SUM):")
    print_tensor("data", tensor, rank, show_all=True)
    
    # Also demonstrate AVERAGE
    tensor = torch.tensor([rank + 1.0], dtype=torch.float32).cuda()
    
    if rank == 0:
        print("\nBefore all_reduce (AVG):")
    print_tensor("data", tensor, rank, show_all=True)
    
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    
    if rank == 0:
        print("\nAfter all_reduce (AVG):")
    print_tensor("data", tensor, rank, show_all=True)


# ============================================================================
# ALL-GATHER: Collect data from all GPUs to all GPUs
# ============================================================================

def demo_all_gather(rank: int, world_size: int):
    """
    All-Gather: Each GPU contributes data, all GPUs get all data.
    
    Used in: FSDP to gather model shards
    
    Example:
        Before: GPU0=[A], GPU1=[B], GPU2=[C], GPU3=[D]
        After:  GPU0=[A,B,C,D], GPU1=[A,B,C,D], GPU2=[A,B,C,D], GPU3=[A,B,C,D]
    """
    if rank == 0:
        print("\n" + "=" * 60)
        print("ALL-GATHER OPERATION")
        print("=" * 60)
        print("Each rank contributes data, all ranks receive all data")
        print("Used by: FSDP to gather parameter shards")
    
    # Each rank has a small tensor
    tensor = torch.tensor([rank * 10 + 1, rank * 10 + 2]).cuda()
    
    # Prepare output list
    output = [torch.zeros(2, dtype=torch.long).cuda() for _ in range(world_size)]
    
    if rank == 0:
        print("\nBefore all_gather:")
    print_tensor("local", tensor, rank, show_all=True)
    
    # All-gather
    dist.all_gather(output, tensor)
    
    if rank == 0:
        print("\nAfter all_gather:")
    
    result = torch.cat(output)
    print_tensor("gathered", result, rank, show_all=True)


# ============================================================================
# REDUCE-SCATTER: Reduce then scatter to different GPUs
# ============================================================================

def demo_reduce_scatter(rank: int, world_size: int):
    """
    Reduce-Scatter: Reduce across ranks, then scatter results.
    
    Used in: FSDP/ZeRO for gradient reduction
    
    Example with 4 GPUs, each has [A,B,C,D]:
        GPU0 gets sum of all A's
        GPU1 gets sum of all B's
        GPU2 gets sum of all C's
        GPU3 gets sum of all D's
    """
    if rank == 0:
        print("\n" + "=" * 60)
        print("REDUCE-SCATTER OPERATION")
        print("=" * 60)
        print("Reduce across ranks, each rank gets a portion")
        print("Used by: FSDP/ZeRO for sharded gradient reduction")
    
    # Each rank has data divisible by world_size
    # Value indicates [rank_contribution, position]
    input_tensor = torch.tensor(
        [rank * 100 + i for i in range(world_size)],
        dtype=torch.float32
    ).cuda()
    
    # Output is 1/world_size of input size
    output = torch.zeros(1, dtype=torch.float32).cuda()
    
    if rank == 0:
        print("\nBefore reduce_scatter:")
    print_tensor("input", input_tensor, rank, show_all=True)
    
    # Reduce-scatter
    dist.reduce_scatter_tensor(output, input_tensor, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        print("\nAfter reduce_scatter (each rank has reduced portion):")
    print_tensor("output", output, rank, show_all=True)


# ============================================================================
# BROADCAST: One rank sends to all
# ============================================================================

def demo_broadcast(rank: int, world_size: int):
    """
    Broadcast: One rank sends data to all other ranks.
    
    Used in: Model initialization, hyperparameter sharing
    
    Example (src=0):
        Before: GPU0=[42], GPU1=[??], GPU2=[??], GPU3=[??]
        After:  GPU0=[42], GPU1=[42], GPU2=[42], GPU3=[42]
    """
    if rank == 0:
        print("\n" + "=" * 60)
        print("BROADCAST OPERATION")
        print("=" * 60)
        print("One rank sends data to all other ranks")
        print("Used by: Model initialization from rank 0")
    
    # Only rank 0 has the data
    if rank == 0:
        tensor = torch.tensor([42, 43, 44, 45]).cuda()
    else:
        tensor = torch.zeros(4, dtype=torch.long).cuda()
    
    if rank == 0:
        print("\nBefore broadcast (src=0):")
    print_tensor("data", tensor, rank, show_all=True)
    
    # Broadcast from rank 0
    dist.broadcast(tensor, src=0)
    
    if rank == 0:
        print("\nAfter broadcast:")
    print_tensor("data", tensor, rank, show_all=True)


# ============================================================================
# SCATTER: One rank distributes different data to each rank
# ============================================================================

def demo_scatter(rank: int, world_size: int):
    """
    Scatter: One rank sends different data to each rank.
    
    Example (src=0):
        Before: GPU0=[A,B,C,D], others=empty
        After:  GPU0=[A], GPU1=[B], GPU2=[C], GPU3=[D]
    """
    if rank == 0:
        print("\n" + "=" * 60)
        print("SCATTER OPERATION")
        print("=" * 60)
        print("One rank distributes different portions to each rank")
    
    # Source rank has all data
    if rank == 0:
        scatter_list = [torch.tensor([i * 10 + 1, i * 10 + 2]).cuda() 
                       for i in range(world_size)]
    else:
        scatter_list = None
    
    output = torch.zeros(2, dtype=torch.long).cuda()
    
    if rank == 0:
        print("\nBefore scatter (src=0):")
        for i, t in enumerate(scatter_list):
            print(f"  To rank {i}: {t.tolist()}")
    
    dist.barrier()
    
    # Scatter
    dist.scatter(output, scatter_list if rank == 0 else None, src=0)
    
    if rank == 0:
        print("\nAfter scatter:")
    print_tensor("received", output, rank, show_all=True)


# ============================================================================
# GATHER: All ranks send to one rank
# ============================================================================

def demo_gather(rank: int, world_size: int):
    """
    Gather: All ranks send data to one rank.
    
    Example (dst=0):
        Before: GPU0=[A], GPU1=[B], GPU2=[C], GPU3=[D]
        After:  GPU0=[A,B,C,D], others unchanged
    """
    if rank == 0:
        print("\n" + "=" * 60)
        print("GATHER OPERATION")
        print("=" * 60)
        print("All ranks send data to destination rank")
    
    # Each rank has its own data
    tensor = torch.tensor([rank * 10 + 1, rank * 10 + 2]).cuda()
    
    # Only destination allocates output
    if rank == 0:
        gather_list = [torch.zeros(2, dtype=torch.long).cuda() 
                      for _ in range(world_size)]
    else:
        gather_list = None
    
    if rank == 0:
        print("\nBefore gather (dst=0):")
    print_tensor("local", tensor, rank, show_all=True)
    
    # Gather
    dist.gather(tensor, gather_list if rank == 0 else None, dst=0)
    
    if rank == 0:
        print("\nAfter gather (only rank 0 has all data):")
        result = torch.cat(gather_list)
        print(f"  [Rank 0] gathered: {result.tolist()}")


# ============================================================================
# SEND/RECV: Point-to-point communication
# ============================================================================

def demo_send_recv(rank: int, world_size: int):
    """
    Send/Recv: Direct communication between two ranks.
    
    Used in: Pipeline parallelism for activation transfer
    """
    if rank == 0:
        print("\n" + "=" * 60)
        print("SEND/RECV (Point-to-Point) OPERATION")
        print("=" * 60)
        print("Direct communication between two specific ranks")
        print("Used by: Pipeline parallelism")
    
    dist.barrier()
    
    # Ring communication: rank i sends to rank (i+1) % world_size
    send_tensor = torch.tensor([rank * 100 + 1, rank * 100 + 2]).cuda()
    recv_tensor = torch.zeros(2, dtype=torch.long).cuda()
    
    send_to = (rank + 1) % world_size
    recv_from = (rank - 1) % world_size
    
    if rank == 0:
        print(f"\nRing communication pattern:")
        for r in range(world_size):
            print(f"  Rank {r} → Rank {(r+1) % world_size}")
    
    dist.barrier()
    
    # Use isend/irecv for non-blocking
    send_req = dist.isend(send_tensor, dst=send_to)
    recv_req = dist.irecv(recv_tensor, src=recv_from)
    
    send_req.wait()
    recv_req.wait()
    
    if rank == 0:
        print("\nAfter ring exchange:")
    print_tensor(f"sent to {send_to}, received from {recv_from}", 
                recv_tensor, rank, show_all=True)


# ============================================================================
# Process Groups
# ============================================================================

def demo_process_groups(rank: int, world_size: int):
    """
    Process Groups: Create sub-groups for partial communication.
    
    Used in: Tensor parallelism, hybrid parallelism
    """
    if world_size < 4:
        if rank == 0:
            print("\nSkipping process groups demo (need 4+ GPUs)")
        return
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("PROCESS GROUPS")
        print("=" * 60)
        print("Create sub-groups for communication within subsets")
        print("Used by: Tensor parallelism, Hybrid parallelism")
    
    # Create two groups: [0,1] and [2,3]
    group1_ranks = list(range(0, world_size // 2))
    group2_ranks = list(range(world_size // 2, world_size))
    
    group1 = dist.new_group(group1_ranks)
    group2 = dist.new_group(group2_ranks)
    
    # Determine which group this rank belongs to
    my_group = group1 if rank < world_size // 2 else group2
    my_group_ranks = group1_ranks if rank < world_size // 2 else group2_ranks
    
    if rank == 0:
        print(f"\nGroup 1: ranks {group1_ranks}")
        print(f"Group 2: ranks {group2_ranks}")
    
    dist.barrier()
    
    # All-reduce within group
    tensor = torch.tensor([rank + 1.0]).cuda()
    
    if rank == 0:
        print("\nBefore intra-group all_reduce:")
    print_tensor("data", tensor, rank, show_all=True)
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=my_group)
    
    if rank == 0:
        print("\nAfter intra-group all_reduce:")
    print_tensor("data", tensor, rank, show_all=True)


# ============================================================================
# Async Operations
# ============================================================================

def demo_async_ops(rank: int, world_size: int):
    """
    Async Operations: Non-blocking collectives for overlap with compute.
    """
    if rank == 0:
        print("\n" + "=" * 60)
        print("ASYNC OPERATIONS")
        print("=" * 60)
        print("Non-blocking collectives to overlap communication with compute")
    
    # Create tensors
    tensor = torch.randn(1000, 1000).cuda() * (rank + 1)
    
    if rank == 0:
        print("\nStarting async all_reduce...")
    
    # Async all-reduce
    work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
    
    # Do some compute while communication happens
    dummy_compute = torch.randn(500, 500).cuda() @ torch.randn(500, 500).cuda()
    
    # Wait for communication to complete
    work.wait()
    
    if rank == 0:
        print("Async all_reduce completed!")
        print(f"Tensor mean (should be sum of (rank+1) for all ranks): "
              f"{tensor.mean().item():.2f}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="NCCL Collective Operations Demo")
    parser.add_argument("--op", type=str, default="all",
                       choices=["all", "allreduce", "allgather", "reducescatter",
                               "broadcast", "scatter", "gather", "sendrecv",
                               "groups", "async"])
    args = parser.parse_args()
    
    local_rank, rank, world_size = setup_distributed()
    
    if rank == 0:
        print("=" * 60)
        print("NCCL COLLECTIVE OPERATIONS DEMO")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Device: {torch.cuda.get_device_name()}")
    
    demos = {
        "allreduce": demo_all_reduce,
        "allgather": demo_all_gather,
        "reducescatter": demo_reduce_scatter,
        "broadcast": demo_broadcast,
        "scatter": demo_scatter,
        "gather": demo_gather,
        "sendrecv": demo_send_recv,
        "groups": demo_process_groups,
        "async": demo_async_ops,
    }
    
    if args.op == "all":
        for name, demo in demos.items():
            try:
                demo(rank, world_size)
            except Exception as e:
                if rank == 0:
                    print(f"Error in {name}: {e}")
    else:
        demos[args.op](rank, world_size)
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
