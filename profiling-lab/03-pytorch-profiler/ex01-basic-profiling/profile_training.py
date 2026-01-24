"""
profile_training.py - Complete PyTorch Profiler example

This script demonstrates comprehensive training loop profiling with:
- Forward/backward/optimizer timing
- Memory profiling
- TensorBoard export
- Data loading analysis

Usage:
    python profile_training.py
    tensorboard --logdir=./profiler_logs

Author: CUDA Lab
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    tensorboard_trace_handler,
)
import argparse
from pathlib import Path


class SimpleModel(nn.Module):
    """Simple model for profiling demonstration."""
    
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )
    
    def forward(self, x):
        return self.layers(x)


def create_dummy_dataset(num_samples=10000, input_size=784, num_classes=10):
    """Create dummy dataset for profiling."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


def profile_basic(model, input_data, device):
    """Basic profiling example."""
    print("\n=== Basic Profiling ===")
    
    model.eval()
    input_data = input_data.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(input_data)
    
    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        with record_function("model_inference"):
            for _ in range(100):
                model(input_data)
    
    # Print results
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=15
    ))
    
    return prof


def profile_training_loop(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    log_dir="./profiler_logs",
    use_amp=False,
):
    """Profile a complete training loop."""
    print(f"\n=== Training Loop Profiling (AMP={use_amp}) ===")
    
    model.train()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    # Profile with schedule
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=2,      # Skip first 2 steps
            warmup=2,    # Warmup for 2 steps
            active=6,    # Profile 6 steps
            repeat=1     # One cycle
        ),
        on_trace_ready=tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        
        for step, (data, target) in enumerate(train_loader):
            if step >= 10:  # Limit steps for demo
                break
            
            with record_function("data_transfer"):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
            
            with record_function("zero_grad"):
                optimizer.zero_grad(set_to_none=True)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    with record_function("forward"):
                        output = model(data)
                    with record_function("loss"):
                        loss = criterion(output, target)
                
                with record_function("backward"):
                    scaler.scale(loss).backward()
                
                with record_function("optimizer"):
                    scaler.step(optimizer)
                    scaler.update()
            else:
                with record_function("forward"):
                    output = model(data)
                
                with record_function("loss"):
                    loss = criterion(output, target)
                
                with record_function("backward"):
                    loss.backward()
                
                with record_function("optimizer"):
                    optimizer.step()
            
            prof.step()  # Signal profiler about step boundary
            
            if step % 5 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
    
    # Print summary
    print("\nTop operations by CUDA time:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    ))
    
    return prof


def profile_memory(model, input_data, device):
    """Profile memory usage."""
    print("\n=== Memory Profiling ===")
    
    model.train()
    input_data = input_data.to(device)
    target = torch.zeros(input_data.size(0), dtype=torch.long, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    with profile(
        activities=[ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        # Forward
        output = model(input_data)
        loss = criterion(output, target)
        
        # Backward
        loss.backward()
        
        # Optimizer step
        optimizer.step()
    
    # Memory statistics
    print("\nMemory usage by operation:")
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=15
    ))
    
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nPeak GPU memory: {peak_memory:.2f} GB")
    
    return prof


def compare_implementations(model, input_data, device):
    """Compare FP32 vs FP16 performance."""
    print("\n=== FP32 vs FP16 Comparison ===")
    
    model.eval()
    input_data = input_data.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(20):
            model(input_data)
    
    # FP32
    with profile(activities=[ProfilerActivity.CUDA]) as prof_fp32:
        with torch.no_grad():
            for _ in range(100):
                model(input_data)
    
    fp32_time = sum(e.cuda_time_total for e in prof_fp32.key_averages()) / 1000
    
    # FP16
    with profile(activities=[ProfilerActivity.CUDA]) as prof_fp16:
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for _ in range(100):
                    model(input_data)
    
    fp16_time = sum(e.cuda_time_total for e in prof_fp16.key_averages()) / 1000
    
    print(f"FP32 total time: {fp32_time:.2f} ms")
    print(f"FP16 total time: {fp16_time:.2f} ms")
    print(f"Speedup: {fp32_time/fp16_time:.2f}x")


def export_traces(prof, output_dir="./traces"):
    """Export profiling traces."""
    print(f"\n=== Exporting Traces to {output_dir} ===")
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Chrome trace format
    chrome_path = f"{output_dir}/trace.json"
    prof.export_chrome_trace(chrome_path)
    print(f"Chrome trace: {chrome_path}")
    print("  View with: chrome://tracing")
    
    # Stack traces
    stack_path = f"{output_dir}/stacks.txt"
    prof.export_stacks(stack_path, "self_cuda_time_total")
    print(f"Stack traces: {stack_path}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch Profiler Demo")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--log-dir', type=str, default='./profiler_logs')
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--export-traces', action='store_true')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create model and data
    model = SimpleModel(hidden_size=args.hidden_size).to(device)
    dataset = create_dummy_dataset()
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Run profiling demos
    input_data = torch.randn(args.batch_size, 784)
    
    # 1. Basic profiling
    prof = profile_basic(model, input_data, device)
    
    # 2. Training loop profiling
    prof = profile_training_loop(
        model, train_loader, criterion, optimizer, device,
        log_dir=args.log_dir,
        use_amp=args.use_amp,
    )
    
    # 3. Memory profiling
    model = SimpleModel(hidden_size=args.hidden_size).to(device)  # Fresh model
    optimizer = optim.Adam(model.parameters())
    profile_memory(model, input_data, device)
    
    # 4. FP32 vs FP16 comparison
    compare_implementations(model, input_data, device)
    
    # 5. Export traces
    if args.export_traces:
        export_traces(prof)
    
    print(f"\n=== Profiling Complete ===")
    print(f"View TensorBoard: tensorboard --logdir={args.log_dir}")


if __name__ == "__main__":
    main()
