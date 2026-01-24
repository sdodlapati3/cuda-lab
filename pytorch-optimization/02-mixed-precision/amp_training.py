"""
amp_training.py - Automatic Mixed Precision Training Examples

Demonstrates:
1. Basic AMP with GradScaler
2. Different precision strategies (FP16, BF16)
3. AMP with gradient accumulation
4. Custom autocast contexts

Author: CUDA Lab
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import time
import argparse
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass 
class TrainingMetrics:
    """Metrics from training run."""
    total_time_s: float
    samples_per_sec: float
    peak_memory_MB: float
    final_loss: float


class SimpleModel(nn.Module):
    """Simple model for AMP testing."""
    
    def __init__(self, input_dim=768, hidden_dim=3072, output_dim=10):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class TransformerModel(nn.Module):
    """Transformer-like model for AMP testing."""
    
    def __init__(self, d_model=768, n_heads=12, n_layers=6, num_classes=10):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)


def train_fp32(
    model: nn.Module,
    dataloader,
    epochs: int,
    device: torch.device
) -> TrainingMetrics:
    """Standard FP32 training."""
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.perf_counter()
    total_samples = 0
    final_loss = 0.0
    
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_samples += inputs.shape[0]
            final_loss = loss.item()
    
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1e6
    
    return TrainingMetrics(
        total_time_s=total_time,
        samples_per_sec=total_samples / total_time,
        peak_memory_MB=peak_memory,
        final_loss=final_loss
    )


def train_amp_fp16(
    model: nn.Module,
    dataloader,
    epochs: int,
    device: torch.device
) -> TrainingMetrics:
    """Mixed precision training with FP16."""
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    model.train()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.perf_counter()
    total_samples = 0
    final_loss = 0.0
    
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Autocast for mixed precision
            with autocast(dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_samples += inputs.shape[0]
            final_loss = loss.item()
    
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1e6
    
    return TrainingMetrics(
        total_time_s=total_time,
        samples_per_sec=total_samples / total_time,
        peak_memory_MB=peak_memory,
        final_loss=final_loss
    )


def train_amp_bf16(
    model: nn.Module,
    dataloader,
    epochs: int,
    device: torch.device
) -> TrainingMetrics:
    """Mixed precision training with BF16 (no scaler needed)."""
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.perf_counter()
    total_samples = 0
    final_loss = 0.0
    
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # BF16 doesn't need scaling
            with autocast(dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            total_samples += inputs.shape[0]
            final_loss = loss.item()
    
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1e6
    
    return TrainingMetrics(
        total_time_s=total_time,
        samples_per_sec=total_samples / total_time,
        peak_memory_MB=peak_memory,
        final_loss=final_loss
    )


def train_with_gradient_accumulation(
    model: nn.Module,
    dataloader,
    epochs: int,
    device: torch.device,
    accumulation_steps: int = 4
) -> TrainingMetrics:
    """AMP with gradient accumulation for larger effective batch sizes."""
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    model.train()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.perf_counter()
    total_samples = 0
    final_loss = 0.0
    
    for epoch in range(epochs):
        accumulated_loss = 0.0
        
        for step, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward with autocast
            with autocast(dtype=torch.float16):
                outputs = model(inputs)
                # Scale loss for accumulation
                loss = criterion(outputs, targets) / accumulation_steps
            
            # Backward (accumulate gradients)
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()
            
            # Update weights every accumulation_steps
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accumulated_loss = 0.0
            
            total_samples += inputs.shape[0]
            final_loss = loss.item() * accumulation_steps
    
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1e6
    
    return TrainingMetrics(
        total_time_s=total_time,
        samples_per_sec=total_samples / total_time,
        peak_memory_MB=peak_memory,
        final_loss=final_loss
    )


class SyntheticDataLoader:
    """Synthetic data loader for benchmarking."""
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        n_batches: int,
        num_classes: int = 10
    ):
        self.input_shape = input_shape
        self.n_batches = n_batches
        self.num_classes = num_classes
    
    def __iter__(self):
        for _ in range(self.n_batches):
            inputs = torch.randn(*self.input_shape)
            targets = torch.randint(0, self.num_classes, (self.input_shape[0],))
            yield inputs, targets
    
    def __len__(self):
        return self.n_batches


def compare_precision_modes(
    model_fn,
    model_name: str,
    input_shape: Tuple[int, ...],
    n_batches: int = 100,
    epochs: int = 1
):
    """Compare FP32, FP16, and BF16 training."""
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Input shape: {input_shape}, {n_batches} batches")
    print(f"{'='*60}")
    
    device = torch.device('cuda')
    dataloader = SyntheticDataLoader(input_shape, n_batches)
    
    results = {}
    
    # FP32 baseline
    print("\nFP32 Training...")
    model = model_fn()
    metrics_fp32 = train_fp32(model, dataloader, epochs, device)
    results['FP32'] = metrics_fp32
    del model
    torch.cuda.empty_cache()
    
    # FP16 AMP
    print("FP16 AMP Training...")
    model = model_fn()
    metrics_fp16 = train_amp_fp16(model, dataloader, epochs, device)
    results['FP16'] = metrics_fp16
    del model
    torch.cuda.empty_cache()
    
    # BF16 (if supported)
    if torch.cuda.is_bf16_supported():
        print("BF16 Training...")
        model = model_fn()
        metrics_bf16 = train_amp_bf16(model, dataloader, epochs, device)
        results['BF16'] = metrics_bf16
        del model
        torch.cuda.empty_cache()
    else:
        print("BF16 not supported on this GPU")
    
    # Print results
    print("\n" + "-"*60)
    print(f"{'Mode':<10} {'Time (s)':<12} {'Samples/s':<12} {'Memory (MB)':<12} {'Speedup':<10}")
    print("-"*60)
    
    baseline_throughput = results['FP32'].samples_per_sec
    
    for mode, metrics in results.items():
        speedup = metrics.samples_per_sec / baseline_throughput
        print(f"{mode:<10} {metrics.total_time_s:<12.2f} {metrics.samples_per_sec:<12.0f} "
              f"{metrics.peak_memory_MB:<12.0f} {speedup:<10.2f}x")
    
    return results


def demonstrate_precision_sensitivity():
    """Show operations that are sensitive to precision."""
    
    print("\n" + "="*60)
    print("Precision Sensitivity Demonstration")
    print("="*60)
    
    device = torch.device('cuda')
    
    # Large values that cause FP16 overflow
    print("\n1. Overflow in FP16:")
    x_large = torch.tensor([65000.0], device=device)
    
    print(f"   FP32: {x_large.float()} * 2 = {(x_large.float() * 2).item()}")
    print(f"   FP16: {x_large.half()} * 2 = {(x_large.half() * 2).item()} (overflow!)")
    
    # Small values that underflow
    print("\n2. Underflow in FP16:")
    x_small = torch.tensor([1e-6], device=device)
    
    print(f"   FP32: {x_small.float()}")
    print(f"   FP16: {x_small.half()} (loss of precision)")
    
    # Softmax stability
    print("\n3. Softmax stability:")
    logits = torch.randn(1, 10, device=device) * 10
    
    with autocast(dtype=torch.float16):
        softmax_fp16 = torch.softmax(logits, dim=-1)
    
    softmax_fp32 = torch.softmax(logits, dim=-1)
    
    diff = (softmax_fp16.float() - softmax_fp32).abs().max()
    print(f"   Max difference: {diff.item():.2e}")
    print("   (softmax is computed in FP32 by autocast for stability)")


def main():
    parser = argparse.ArgumentParser(description='AMP Training Examples')
    parser.add_argument('--model', type=str, default='simple',
                       choices=['simple', 'transformer'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seq-length', type=int, default=256)
    parser.add_argument('--n-batches', type=int, default=100)
    parser.add_argument('--sensitivity-demo', action='store_true',
                       help='Show precision sensitivity examples')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA required for AMP")
        return
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    
    if args.sensitivity_demo:
        demonstrate_precision_sensitivity()
        return
    
    # Select model
    if args.model == 'simple':
        model_fn = lambda: SimpleModel()
        input_shape = (args.batch_size, 768)
    else:
        model_fn = lambda: TransformerModel()
        input_shape = (args.batch_size, args.seq_length, 768)
    
    # Run comparison
    compare_precision_modes(
        model_fn,
        args.model,
        input_shape,
        n_batches=args.n_batches
    )


if __name__ == "__main__":
    main()
