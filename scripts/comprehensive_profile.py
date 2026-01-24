#!/usr/bin/env python
"""
Comprehensive GPU Profiling Script for Nsight Systems

This script demonstrates various CUDA operations for profiling:
- Matrix multiplication (GEMM)
- Softmax operations
- Attention mechanisms
- Training loop with forward/backward passes
- Memory transfers (CPU <-> GPU)
- NVTX markers for easy navigation

Usage:
    nsys profile --trace=cuda,cudnn,cublas,nvtx --cuda-memory-usage=true \
        -o training_profile python scripts/comprehensive_profile.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# NVTX for custom markers (shows as colored ranges in Nsight)
try:
    import torch.cuda.nvtx as nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False
    class nvtx:
        @staticmethod
        def range_push(name): pass
        @staticmethod
        def range_pop(): pass


def print_gpu_info():
    """Print GPU information."""
    print("=" * 60)
    print("GPU PROFILING SESSION")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NVTX Available: {HAS_NVTX}")
    print("=" * 60)
    print()


class TransformerBlock(nn.Module):
    """Simple transformer block for profiling attention + FFN."""
    
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Multi-head attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, S, D = x.shape
        
        # Self-attention with NVTX marker
        nvtx.range_push("Self-Attention")
        
        residual = x
        x = self.norm1(x)
        
        # Project Q, K, V
        nvtx.range_push("QKV Projection")
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        nvtx.range_pop()
        
        # Scaled dot-product attention (uses Flash Attention on H100)
        nvtx.range_push("SDPA (Flash Attention)")
        attn_out = F.scaled_dot_product_attention(q, k, v)
        nvtx.range_pop()
        
        # Output projection
        nvtx.range_push("Output Projection")
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        attn_out = self.out_proj(attn_out)
        x = residual + self.dropout(attn_out)
        nvtx.range_pop()
        
        nvtx.range_pop()  # End Self-Attention
        
        # Feed-forward network with NVTX marker
        nvtx.range_push("Feed-Forward Network")
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        nvtx.range_pop()
        
        return x


class SimpleTransformer(nn.Module):
    """Simple transformer model for profiling."""
    
    def __init__(self, vocab_size=10000, d_model=512, n_layers=4, n_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        nvtx.range_push("Embedding")
        x = self.embedding(x)
        nvtx.range_pop()
        
        for i, layer in enumerate(self.layers):
            nvtx.range_push(f"Transformer Layer {i}")
            x = layer(x)
            nvtx.range_pop()
        
        nvtx.range_push("Output Projection")
        x = self.output(x)
        nvtx.range_pop()
        
        return x


def profile_memory_transfers():
    """Profile CPU <-> GPU memory transfers."""
    print("1. Profiling Memory Transfers...")
    
    nvtx.range_push("Memory Transfers")
    
    # CPU -> GPU transfer
    nvtx.range_push("CPU to GPU (pinned)")
    cpu_tensor = torch.randn(1024, 1024, pin_memory=True)
    gpu_tensor = cpu_tensor.cuda(non_blocking=True)
    torch.cuda.synchronize()
    nvtx.range_pop()
    
    # GPU computation
    nvtx.range_push("GPU Compute")
    result = torch.mm(gpu_tensor, gpu_tensor.T)
    torch.cuda.synchronize()
    nvtx.range_pop()
    
    # GPU -> CPU transfer
    nvtx.range_push("GPU to CPU")
    cpu_result = result.cpu()
    nvtx.range_pop()
    
    nvtx.range_pop()  # End Memory Transfers
    
    print(f"   Transferred: {cpu_tensor.numel() * 4 / 1e6:.1f} MB")
    print()


def profile_matmul_sizes():
    """Profile matrix multiplications of various sizes."""
    print("2. Profiling Matrix Multiplications...")
    
    nvtx.range_push("MatMul Benchmarks")
    
    sizes = [512, 1024, 2048, 4096]
    
    for size in sizes:
        nvtx.range_push(f"MatMul {size}x{size}")
        
        A = torch.randn(size, size, device='cuda')
        B = torch.randn(size, size, device='cuda')
        torch.cuda.synchronize()
        
        # Warmup
        _ = torch.mm(A, B)
        torch.cuda.synchronize()
        
        # Timed run
        nvtx.range_push("Compute")
        for _ in range(5):
            C = torch.mm(A, B)
        torch.cuda.synchronize()
        nvtx.range_pop()
        
        nvtx.range_pop()
        
        flops = 2 * size**3 * 5
        print(f"   {size}x{size}: {flops/1e12:.2f} TFLOPs total")
    
    nvtx.range_pop()  # End MatMul Benchmarks
    print()


def profile_softmax():
    """Profile softmax operations."""
    print("3. Profiling Softmax Operations...")
    
    nvtx.range_push("Softmax Benchmarks")
    
    batch_sizes = [32, 64]
    seq_lengths = [512, 1024, 2048]
    
    for bs in batch_sizes:
        for seq_len in seq_lengths:
            nvtx.range_push(f"Softmax B={bs} S={seq_len}")
            
            x = torch.randn(bs, seq_len, 768, device='cuda')
            torch.cuda.synchronize()
            
            for _ in range(10):
                y = F.softmax(x, dim=-1)
            torch.cuda.synchronize()
            
            nvtx.range_pop()
    
    nvtx.range_pop()
    print("   Completed softmax benchmarks")
    print()


def profile_attention():
    """Profile attention mechanisms."""
    print("4. Profiling Attention Mechanisms...")
    
    nvtx.range_push("Attention Benchmarks")
    
    configs = [
        (8, 512, 8, 64),   # batch, seq, heads, head_dim
        (8, 1024, 8, 64),
        (4, 2048, 8, 64),
    ]
    
    for batch, seq, heads, head_dim in configs:
        nvtx.range_push(f"Attention B={batch} S={seq}")
        
        q = torch.randn(batch, heads, seq, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch, heads, seq, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(batch, heads, seq, head_dim, device='cuda', dtype=torch.float16)
        torch.cuda.synchronize()
        
        # Standard attention
        nvtx.range_push("Standard Attention")
        scale = head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        out_standard = torch.matmul(attn, v)
        torch.cuda.synchronize()
        nvtx.range_pop()
        
        # Flash attention (SDPA)
        nvtx.range_push("Flash Attention (SDPA)")
        out_flash = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        nvtx.range_pop()
        
        nvtx.range_pop()
    
    nvtx.range_pop()
    print("   Completed attention benchmarks")
    print()


def profile_training_loop():
    """Profile a complete training loop."""
    print("5. Profiling Training Loop...")
    
    nvtx.range_push("Training Loop")
    
    # Model setup
    nvtx.range_push("Model Setup")
    model = SimpleTransformer(
        vocab_size=10000,
        d_model=512,
        n_layers=4,
        n_heads=8
    ).cuda()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    nvtx.range_pop()
    
    # Training iterations
    batch_size = 16
    seq_length = 256
    num_steps = 5
    
    for step in range(num_steps):
        nvtx.range_push(f"Training Step {step}")
        
        # Generate batch
        nvtx.range_push("Data Generation")
        inputs = torch.randint(0, 10000, (batch_size, seq_length), device='cuda')
        targets = torch.randint(0, 10000, (batch_size, seq_length), device='cuda')
        nvtx.range_pop()
        
        # Forward pass
        nvtx.range_push("Forward Pass")
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, 10000), targets.view(-1))
        nvtx.range_pop()
        
        # Backward pass
        nvtx.range_push("Backward Pass")
        optimizer.zero_grad()
        loss.backward()
        nvtx.range_pop()
        
        # Optimizer step
        nvtx.range_push("Optimizer Step")
        optimizer.step()
        torch.cuda.synchronize()
        nvtx.range_pop()
        
        nvtx.range_pop()  # End Training Step
        
        print(f"   Step {step}: Loss = {loss.item():.4f}")
    
    nvtx.range_pop()  # End Training Loop
    print()


def profile_mixed_precision():
    """Profile mixed precision training."""
    print("6. Profiling Mixed Precision Training...")
    
    nvtx.range_push("Mixed Precision Training")
    
    model = SimpleTransformer(
        vocab_size=10000,
        d_model=512,
        n_layers=2,
        n_heads=8
    ).cuda()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 32
    seq_length = 512
    
    for step in range(3):
        nvtx.range_push(f"AMP Step {step}")
        
        inputs = torch.randint(0, 10000, (batch_size, seq_length), device='cuda')
        targets = torch.randint(0, 10000, (batch_size, seq_length), device='cuda')
        
        # Forward with autocast
        nvtx.range_push("AMP Forward")
        with torch.amp.autocast('cuda', dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, 10000), targets.view(-1))
        nvtx.range_pop()
        
        # Scaled backward
        nvtx.range_push("AMP Backward")
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        nvtx.range_pop()
        
        # Scaled optimizer step
        nvtx.range_push("AMP Optimizer")
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        nvtx.range_pop()
        
        nvtx.range_pop()
        
        print(f"   AMP Step {step}: Loss = {loss.item():.4f}")
    
    nvtx.range_pop()
    print()


def main():
    print_gpu_info()
    
    # Warmup GPU
    nvtx.range_push("GPU Warmup")
    _ = torch.randn(1000, 1000, device='cuda') @ torch.randn(1000, 1000, device='cuda')
    torch.cuda.synchronize()
    nvtx.range_pop()
    
    # Run all profiling sections
    profile_memory_transfers()
    profile_matmul_sizes()
    profile_softmax()
    profile_attention()
    profile_training_loop()
    profile_mixed_precision()
    
    # Final sync
    torch.cuda.synchronize()
    
    print("=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)
    print()
    print("Open the .nsys-rep file in Nsight Systems GUI to visualize:")
    print("  - Timeline with NVTX markers (colored ranges)")
    print("  - CUDA kernel execution")
    print("  - Memory transfers")
    print("  - CPU/GPU overlap")


if __name__ == "__main__":
    main()
