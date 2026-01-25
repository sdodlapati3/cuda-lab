#!/usr/bin/env python3
"""
zero_offload.py - DeepSpeed ZeRO-Offload example for training large models

Demonstrates:
- CPU offloading for optimizer states and parameters
- NVMe offloading for extreme model sizes
- Memory-efficient training on limited GPU resources
- Configuration options for different offload scenarios

Usage:
    # CPU offload
    deepspeed --num_gpus=4 zero_offload.py --offload cpu
    
    # NVMe offload (requires fast SSD)
    deepspeed --num_gpus=4 zero_offload.py --offload nvme --nvme_path /local/nvme

Author: CUDA Lab
"""

import os
import sys
import gc
import time
import argparse
import json
from typing import Dict, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("DeepSpeed not installed. Run: pip install deepspeed")


@dataclass 
class OffloadConfig:
    """Configuration for offload training."""
    # Model size (simulating a 7B model)
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    intermediate_size: int = 11008
    
    # Offload settings
    offload_device: str = "cpu"  # cpu or nvme
    nvme_path: str = "/local/nvme"
    offload_params: bool = True
    offload_optimizer: bool = True
    pin_memory: bool = True
    
    # Training
    batch_size: int = 1
    gradient_accumulation: int = 8
    learning_rate: float = 1e-5
    max_steps: int = 100
    seq_length: int = 512


# ============================================================================
# Model (Same as deepspeed_train.py but larger)
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.attention = Attention(hidden_size, num_heads)
        self.mlp = MLP(hidden_size, intermediate_size)
        self.input_norm = RMSNorm(hidden_size)
        self.post_attn_norm = RMSNorm(hidden_size)
    
    def forward(self, x):
        x = x + self.attention(self.input_norm(x))
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class LargeLanguageModel(nn.Module):
    """Large LLM for offload demonstration."""
    def __init__(self, config: OffloadConfig):
        super().__init__()
        self.config = config
        
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config.hidden_size, config.num_heads, config.intermediate_size)
            for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.embed.weight
    
    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return {"loss": loss, "logits": logits}


# ============================================================================
# DeepSpeed Configuration Generators
# ============================================================================

def create_cpu_offload_config(config: OffloadConfig) -> Dict:
    """Create DeepSpeed config for CPU offloading."""
    return {
        "train_batch_size": config.batch_size * config.gradient_accumulation,
        "train_micro_batch_size_per_gpu": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation,
        "gradient_clipping": 1.0,
        
        "zero_optimization": {
            "stage": 3,
            "offload_param": {
                "device": "cpu",
                "pin_memory": config.pin_memory,
                "buffer_count": 5,
                "buffer_size": 1e8,
                "max_in_cpu": 1e9
            },
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": config.pin_memory,
                "buffer_count": 4,
                "fast_init": False
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        
        "bf16": {
            "enabled": True
        },
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.learning_rate,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": config.learning_rate,
                "warmup_num_steps": 10,
                "total_num_steps": config.max_steps
            }
        },
        
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": True,
            "synchronize_checkpoint_boundary": False
        },
        
        "steps_per_print": 10,
        "wall_clock_breakdown": False
    }


def create_nvme_offload_config(config: OffloadConfig) -> Dict:
    """Create DeepSpeed config for NVMe offloading."""
    ds_config = create_cpu_offload_config(config)
    
    # Modify for NVMe
    ds_config["zero_optimization"]["offload_param"] = {
        "device": "nvme",
        "nvme_path": config.nvme_path,
        "pin_memory": True,
        "buffer_count": 5,
        "buffer_size": 1e9,
        "max_in_cpu": 1e9
    }
    
    ds_config["zero_optimization"]["offload_optimizer"] = {
        "device": "nvme",
        "nvme_path": config.nvme_path,
        "pin_memory": True,
        "buffer_count": 4,
        "fast_init": False
    }
    
    # Add aio config for NVMe
    ds_config["aio"] = {
        "block_size": 1048576,
        "queue_depth": 8,
        "thread_count": 1,
        "single_submit": False,
        "overlap_events": True
    }
    
    return ds_config


# ============================================================================
# Dataset
# ============================================================================

class SyntheticDataset(Dataset):
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        return {"input_ids": tokens, "labels": tokens.clone()}


# ============================================================================
# Training
# ============================================================================

def print_memory_stats(prefix: str = ""):
    """Print GPU and CPU memory statistics."""
    gpu_allocated = torch.cuda.memory_allocated() / 1e9
    gpu_reserved = torch.cuda.memory_reserved() / 1e9
    gpu_max = torch.cuda.max_memory_allocated() / 1e9
    
    import psutil
    cpu_percent = psutil.virtual_memory().percent
    cpu_used = psutil.virtual_memory().used / 1e9
    
    print(f"{prefix} GPU: {gpu_allocated:.2f}GB alloc, {gpu_max:.2f}GB peak | "
          f"CPU: {cpu_used:.1f}GB ({cpu_percent:.0f}%)")


def train(model_engine, train_dataloader, config: OffloadConfig, local_rank: int):
    """Training loop with offloading."""
    model_engine.train()
    
    global_step = 0
    total_loss = 0.0
    start_time = time.time()
    
    if local_rank == 0:
        print("\nStarting training with offloading...")
        print_memory_stats("Initial: ")
    
    for step, batch in enumerate(train_dataloader):
        if global_step >= config.max_steps:
            break
        
        input_ids = batch["input_ids"].to(model_engine.device)
        labels = batch["labels"].to(model_engine.device)
        
        outputs = model_engine(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        
        model_engine.backward(loss)
        model_engine.step()
        
        total_loss += loss.item()
        global_step += 1
        
        if local_rank == 0 and global_step % 10 == 0:
            avg_loss = total_loss / 10
            elapsed = time.time() - start_time
            steps_per_sec = global_step / elapsed
            
            print(f"\nStep {global_step}/{config.max_steps}: "
                  f"loss={avg_loss:.4f}, steps/s={steps_per_sec:.2f}")
            print_memory_stats("Memory: ")
            
            total_loss = 0.0
    
    if local_rank == 0:
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time:.1f}s")
        print(f"Average time per step: {total_time / global_step:.2f}s")
        print_memory_stats("Final: ")


def estimate_model_memory(config: OffloadConfig) -> Dict:
    """Estimate memory requirements."""
    # Parameter count
    embed_params = config.vocab_size * config.hidden_size
    layer_params = (
        4 * config.hidden_size * config.hidden_size +  # attention
        3 * config.hidden_size * config.intermediate_size +  # mlp
        2 * config.hidden_size  # norms
    )
    total_params = embed_params + config.num_layers * layer_params
    
    # Memory estimates (in GB)
    param_memory = total_params * 2 / 1e9  # BF16
    optim_memory = total_params * 12 / 1e9  # Adam states in FP32
    grad_memory = total_params * 2 / 1e9  # BF16 gradients
    
    return {
        "total_params_b": total_params / 1e9,
        "param_memory_gb": param_memory,
        "optimizer_memory_gb": optim_memory,
        "gradient_memory_gb": grad_memory,
        "total_memory_gb": param_memory + optim_memory + grad_memory,
    }


def main():
    if not DEEPSPEED_AVAILABLE:
        print("Please install DeepSpeed: pip install deepspeed")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="DeepSpeed ZeRO-Offload Training")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--offload", type=str, default="cpu", choices=["cpu", "nvme"])
    parser.add_argument("--nvme_path", type=str, default="/local/nvme")
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100)
    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    # Create config
    config = OffloadConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_heads=args.hidden_size // 128,
        intermediate_size=int(args.hidden_size * 2.75),
        offload_device=args.offload,
        nvme_path=args.nvme_path,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
    )
    
    # Create DeepSpeed config
    if args.offload == "cpu":
        ds_config = create_cpu_offload_config(config)
    else:
        ds_config = create_nvme_offload_config(config)
    
    # Initialize
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if local_rank == 0:
        print("=" * 60)
        print("DEEPSPEED ZERO-OFFLOAD TRAINING")
        print("=" * 60)
        print(f"Offload device: {args.offload}")
        if args.offload == "nvme":
            print(f"NVMe path: {args.nvme_path}")
        print(f"Model: {config.num_layers} layers, hidden={config.hidden_size}")
        
        mem_est = estimate_model_memory(config)
        print(f"\nModel size: {mem_est['total_params_b']:.2f}B parameters")
        print(f"Without offload would need: {mem_est['total_memory_gb']:.1f}GB per GPU")
        print("=" * 60)
    
    # Create model
    if local_rank == 0:
        print("\nCreating model (this may take a while for large models)...")
    
    model = LargeLanguageModel(config)
    
    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model created: {total_params / 1e9:.2f}B parameters")
    
    # Create dataset
    dataset = SyntheticDataset(
        config.vocab_size,
        config.seq_length,
        num_samples=config.max_steps * config.batch_size * config.gradient_accumulation * 2,
    )
    
    # Initialize DeepSpeed
    if local_rank == 0:
        print("\nInitializing DeepSpeed with offloading...")
    
    model_engine, _, train_dataloader, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        config=ds_config,
    )
    
    if local_rank == 0:
        print(f"DeepSpeed initialized!")
        print(f"World size: {model_engine.world_size}")
        print_memory_stats("After init: ")
        print("=" * 60)
    
    # Train
    train(model_engine, train_dataloader, config, local_rank)
    
    if local_rank == 0:
        print("\nTraining complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()
