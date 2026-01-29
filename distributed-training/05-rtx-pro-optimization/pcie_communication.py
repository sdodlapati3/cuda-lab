"""
PCIe vs NVLink Communication Strategies
=======================================

This module implements communication optimizations specifically for 
PCIe-connected GPUs (like RTX Pro 6000 workstations) to achieve 
performance closer to NVLink-connected systems.

Bandwidth Comparison:
- NVLink 5 (B200): 1.8 TB/s bidirectional
- PCIe Gen5 x16: 128 GB/s bidirectional (14x slower)

Strategies to compensate:
1. Gradient compression (10-100x reduction)
2. Computation-communication overlap
3. Hierarchical all-reduce
4. Delayed gradient synchronization
"""

import torch
import torch.distributed as dist
from torch import Tensor
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import math


@dataclass
class CommConfig:
    """Communication configuration for PCIe systems"""
    
    # Compression
    use_compression: bool = True
    compression_type: str = "topk"  # topk, random_k, powersgd
    compression_ratio: float = 0.01  # 1% of gradients
    
    # PowerSGD specific
    powersgd_rank: int = 4
    powersgd_warmup_steps: int = 1000
    
    # Overlap
    bucket_cap_mb: float = 25.0
    
    # Local SGD (reduce sync frequency)
    local_sgd_steps: int = 1  # 1 = every step, 4 = every 4 steps


class PowerSGDCompressor:
    """
    PowerSGD: Low-rank gradient compression
    
    Achieves 100-1000x compression with minimal accuracy loss.
    Particularly effective for large models on PCIe systems.
    
    Paper: "PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization"
    """
    
    def __init__(self, rank: int = 4, warmup_steps: int = 1000):
        self.rank = rank
        self.warmup_steps = warmup_steps
        self.step = 0
        self.P_memory: Dict[str, Tensor] = {}  # Memory for P matrices
        self.Q_memory: Dict[str, Tensor] = {}  # Memory for Q matrices
        
    def compress(self, grad: Tensor, name: str) -> Tuple[Tensor, Tensor]:
        """
        Compress gradient using low-rank approximation: G ≈ P @ Q^T
        
        Communication: O(rank * (m + n)) instead of O(m * n)
        For 1024x1024 matrix with rank=4: 8KB vs 4MB = 500x compression
        """
        if grad.dim() == 1:
            # Skip compression for 1D tensors (biases)
            return grad, None
            
        # Reshape to 2D
        orig_shape = grad.shape
        if grad.dim() > 2:
            grad = grad.view(grad.shape[0], -1)
        
        m, n = grad.shape
        
        # Initialize or retrieve memory
        if name not in self.P_memory:
            self.P_memory[name] = torch.randn(m, self.rank, device=grad.device)
            self.Q_memory[name] = torch.randn(n, self.rank, device=grad.device)
        
        P = self.P_memory[name]
        Q = self.Q_memory[name]
        
        # Power iteration step
        # Q = G^T @ P, then orthogonalize
        Q_new = grad.t() @ P
        Q_new, _ = torch.linalg.qr(Q_new)
        
        # P = G @ Q
        P_new = grad @ Q_new
        
        # Update memory for next iteration
        self.P_memory[name] = P_new.clone()
        self.Q_memory[name] = Q_new.clone()
        
        self.step += 1
        
        return P_new, Q_new
    
    def decompress(self, P: Tensor, Q: Optional[Tensor], orig_shape) -> Tensor:
        """Reconstruct gradient from low-rank factors"""
        if Q is None:
            return P
        
        grad = P @ Q.t()
        return grad.view(orig_shape)


class TopKCompressor:
    """
    Top-K sparsification with error feedback
    
    Keeps only top K% of gradient values, accumulates errors
    for correction in subsequent iterations.
    """
    
    def __init__(self, ratio: float = 0.01):
        self.ratio = ratio
        self.error_dict: Dict[str, Tensor] = {}
    
    def compress(self, grad: Tensor, name: str) -> Tuple[Tensor, Tensor, torch.Size]:
        """Compress using top-k sparsification"""
        flat = grad.view(-1)
        
        # Add accumulated error
        if name in self.error_dict:
            flat = flat + self.error_dict[name]
        
        k = max(1, int(flat.numel() * self.ratio))
        
        # Get top-k
        _, indices = torch.topk(flat.abs(), k)
        values = flat[indices]
        
        # Store error
        error = flat.clone()
        error[indices] = 0
        self.error_dict[name] = error
        
        return values, indices, grad.shape
    
    def decompress(self, values: Tensor, indices: Tensor, shape: torch.Size) -> Tensor:
        """Reconstruct from sparse representation"""
        flat = torch.zeros(shape.numel(), device=values.device, dtype=values.dtype)
        flat[indices] = values
        return flat.view(shape)


class HierarchicalAllReduce:
    """
    Hierarchical all-reduce for multi-node PCIe systems
    
    Strategy:
    1. Reduce within node (fast, uses PCIe switch)
    2. All-reduce across nodes (slower, uses network)
    3. Broadcast within node
    
    Reduces inter-node traffic by world_size_per_node factor.
    """
    
    def __init__(self):
        self.local_rank = dist.get_rank() % torch.cuda.device_count()
        self.world_size = dist.get_world_size()
        self.local_size = torch.cuda.device_count()
        self.node_rank = dist.get_rank() // self.local_size
        self.num_nodes = self.world_size // self.local_size
        
        # Create process groups
        self._create_groups()
    
    def _create_groups(self):
        """Create intra-node and inter-node process groups"""
        # Intra-node groups (GPUs on same machine)
        for node in range(self.num_nodes):
            ranks = list(range(node * self.local_size, (node + 1) * self.local_size))
            group = dist.new_group(ranks)
            if self.node_rank == node:
                self.intra_node_group = group
        
        # Inter-node group (one GPU per node)
        leader_ranks = [i * self.local_size for i in range(self.num_nodes)]
        self.inter_node_group = dist.new_group(leader_ranks)
        self.is_node_leader = (self.local_rank == 0)
    
    def all_reduce(self, tensor: Tensor) -> Tensor:
        """Perform hierarchical all-reduce"""
        
        # Step 1: Reduce within node to leader
        dist.reduce(tensor, dst=self.node_rank * self.local_size, 
                   group=self.intra_node_group)
        
        # Step 2: All-reduce among node leaders
        if self.is_node_leader:
            dist.all_reduce(tensor, group=self.inter_node_group)
        
        # Step 3: Broadcast from leader to all GPUs in node
        dist.broadcast(tensor, src=self.node_rank * self.local_size,
                      group=self.intra_node_group)
        
        return tensor


class OverlappedCommunicator:
    """
    Overlap gradient communication with backward pass
    
    Uses bucketing to start all-reduce as soon as enough
    gradients are accumulated, hiding latency behind computation.
    """
    
    def __init__(self, bucket_cap_mb: float = 25.0, compressor=None):
        self.bucket_cap_bytes = int(bucket_cap_mb * 1024 * 1024)
        self.compressor = compressor
        
        self.buckets: List[List[Tuple[str, Tensor]]] = [[]]
        self.bucket_sizes: List[int] = [0]
        self.handles: List = []
        self.compressed_data: List = []
        
    def add_gradient(self, name: str, grad: Tensor):
        """Add gradient to current bucket"""
        grad_bytes = grad.numel() * grad.element_size()
        
        # Start new bucket if current one is full
        if self.bucket_sizes[-1] + grad_bytes > self.bucket_cap_bytes:
            self._flush_bucket()
            self.buckets.append([])
            self.bucket_sizes.append(0)
        
        self.buckets[-1].append((name, grad))
        self.bucket_sizes[-1] += grad_bytes
    
    def _flush_bucket(self):
        """Initiate async all-reduce for current bucket"""
        if not self.buckets[-1]:
            return
        
        # Flatten bucket
        flat_grads = []
        for name, grad in self.buckets[-1]:
            flat_grads.append(grad.view(-1))
        
        bucket_tensor = torch.cat(flat_grads)
        
        if self.compressor:
            # Compress before sending
            compressed = self.compressor.compress(bucket_tensor, f"bucket_{len(self.buckets)}")
            self.compressed_data.append((compressed, [g.shape for _, g in self.buckets[-1]]))
            
            if isinstance(compressed, tuple) and len(compressed) == 2:
                # PowerSGD: all-reduce P and Q separately
                P, Q = compressed
                handle_p = dist.all_reduce(P, async_op=True)
                handle_q = dist.all_reduce(Q, async_op=True) if Q is not None else None
                self.handles.append((handle_p, handle_q))
            else:
                # TopK: all-reduce values only
                values, indices, shape = compressed
                handle = dist.all_reduce(values, async_op=True)
                self.handles.append((handle, None))
        else:
            handle = dist.all_reduce(bucket_tensor, async_op=True)
            self.handles.append((handle, bucket_tensor, [g.shape for _, g in self.buckets[-1]]))
    
    def synchronize(self) -> List[Tensor]:
        """Wait for all async ops and return decompressed gradients"""
        self._flush_bucket()  # Flush last bucket
        
        results = []
        for handle_data in self.handles:
            if self.compressor:
                handle_p, handle_q = handle_data
                handle_p.wait()
                if handle_q:
                    handle_q.wait()
            else:
                handle, tensor, shapes = handle_data
                handle.wait()
                tensor /= dist.get_world_size()
                
                # Unflatten
                offset = 0
                for shape in shapes:
                    size = math.prod(shape)
                    results.append(tensor[offset:offset+size].view(shape))
                    offset += size
        
        # Reset for next iteration
        self.buckets = [[]]
        self.bucket_sizes = [0]
        self.handles = []
        self.compressed_data = []
        
        return results


class LocalSGD:
    """
    Local SGD: Reduce synchronization frequency
    
    Instead of all-reduce every step, train locally for H steps
    then synchronize. Reduces communication by factor of H.
    
    Works well when:
    - Communication is the bottleneck (PCIe systems)
    - Using large batch sizes
    - Model has some noise tolerance
    """
    
    def __init__(self, sync_period: int = 4):
        self.sync_period = sync_period
        self.step = 0
        
    def should_sync(self) -> bool:
        """Check if we should synchronize this step"""
        self.step += 1
        return self.step % self.sync_period == 0
    
    def sync_parameters(self, model):
        """Average parameters across all workers"""
        for param in model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.AVG)


# ============================================================================
# Benchmark: Compare communication strategies
# ============================================================================

def benchmark_strategies():
    """Compare different communication strategies"""
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    
    # Simulate gradient tensor (100M parameters = 400MB in FP32)
    grad = torch.randn(100_000_000, device=device)
    
    strategies = {
        "baseline": lambda g: dist.all_reduce(g.clone()),
        "topk_1%": lambda g: TopKCompressor(0.01).compress(g.clone(), "test"),
        "topk_0.1%": lambda g: TopKCompressor(0.001).compress(g.clone(), "test"),
        "powersgd_r4": lambda g: PowerSGDCompressor(rank=4).compress(g.clone().view(10000, 10000), "test"),
    }
    
    # Warmup
    for _ in range(3):
        dist.all_reduce(grad.clone())
    torch.cuda.synchronize()
    
    # Benchmark
    import time
    
    for name, strategy in strategies.items():
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(10):
            strategy(grad)
            torch.cuda.synchronize()
        
        elapsed = (time.perf_counter() - start) / 10
        
        if rank == 0:
            bandwidth = grad.numel() * 4 / elapsed / 1e9  # GB/s
            print(f"{name}: {elapsed*1000:.2f}ms, effective BW: {bandwidth:.2f} GB/s")


if __name__ == "__main__":
    benchmark_strategies()
