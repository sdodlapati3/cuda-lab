#!/usr/bin/env python3
"""
deepspeed_train.py - Basic DeepSpeed training example

Demonstrates:
- DeepSpeed initialization and configuration
- ZeRO optimization stages
- Mixed precision training
- Gradient accumulation
- Checkpointing

Usage:
    # Single node with 4 GPUs, ZeRO-2
    deepspeed --num_gpus=4 deepspeed_train.py \
        --deepspeed_config configs/zero2_config.json

    # Multi-node (create hostfile first)
    deepspeed --hostfile hostfile.txt deepspeed_train.py \
        --deepspeed_config configs/zero3_config.json

Author: CUDA Lab
"""

import os
import sys
import time
import argparse
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("DeepSpeed not installed. Run: pip install deepspeed")


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    intermediate_size: int = 8192
    
    # Training
    batch_size_per_gpu: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 1000
    
    # Data
    seq_length: int = 512
    num_samples: int = 10000
    
    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"


# ============================================================================
# Model Definition
# ============================================================================

class RMSNorm(nn.Module):
    """RMSNorm layer."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class Attention(nn.Module):
    """Multi-head attention."""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(self, x):
        B, T, C = x.shape
        
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class MLP(nn.Module):
    """Feed-forward with SwiGLU."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single transformer layer."""
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.attention = Attention(config.hidden_size, config.num_heads)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_norm = RMSNorm(config.hidden_size)
        self.post_attn_norm = RMSNorm(config.hidden_size)
    
    def forward(self, x):
        x = x + self.attention(self.input_norm(x))
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class GPTModel(nn.Module):
    """GPT-style language model."""
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)
    
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
# Dataset
# ============================================================================

class SyntheticDataset(Dataset):
    """Synthetic language modeling dataset."""
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
# Training Functions
# ============================================================================

def create_deepspeed_config(config: TrainingConfig, ds_config_path: str) -> Dict:
    """Load and update DeepSpeed config."""
    with open(ds_config_path, "r") as f:
        ds_config = json.load(f)
    
    # Update with training config
    ds_config["train_micro_batch_size_per_gpu"] = config.batch_size_per_gpu
    ds_config["gradient_accumulation_steps"] = config.gradient_accumulation_steps
    
    if ds_config.get("optimizer", {}).get("params", {}).get("lr") == "auto":
        ds_config["optimizer"]["params"]["lr"] = config.learning_rate
    if ds_config.get("optimizer", {}).get("params", {}).get("weight_decay") == "auto":
        ds_config["optimizer"]["params"]["weight_decay"] = config.weight_decay
    
    if ds_config.get("scheduler", {}).get("params", {}).get("warmup_num_steps") == "auto":
        ds_config["scheduler"]["params"]["warmup_num_steps"] = config.warmup_steps
    if ds_config.get("scheduler", {}).get("params", {}).get("warmup_max_lr") == "auto":
        ds_config["scheduler"]["params"]["warmup_max_lr"] = config.learning_rate
    if ds_config.get("scheduler", {}).get("params", {}).get("total_num_steps") == "auto":
        ds_config["scheduler"]["params"]["total_num_steps"] = config.max_steps
    
    return ds_config


def get_model_params_info(model: nn.Module) -> Dict:
    """Get model parameter information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_b": total_params / 1e9,
        "trainable_params_b": trainable_params / 1e9,
    }


def train(
    model_engine: 'DeepSpeedEngine',
    train_dataloader: DataLoader,
    config: TrainingConfig,
    local_rank: int,
):
    """Training loop."""
    model_engine.train()
    
    global_step = 0
    total_loss = 0.0
    start_time = time.time()
    
    while global_step < config.max_steps:
        for batch in train_dataloader:
            if global_step >= config.max_steps:
                break
            
            # Move to device
            input_ids = batch["input_ids"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)
            
            # Forward pass
            outputs = model_engine(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            
            # Backward pass (DeepSpeed handles gradient accumulation)
            model_engine.backward(loss)
            
            # Optimizer step (DeepSpeed handles this)
            model_engine.step()
            
            total_loss += loss.item()
            global_step += 1
            
            # Logging
            if global_step % 10 == 0 and local_rank == 0:
                avg_loss = total_loss / 10
                elapsed = time.time() - start_time
                samples_per_sec = (global_step * config.batch_size_per_gpu * 
                                   config.gradient_accumulation_steps) / elapsed
                
                print(f"Step {global_step}/{config.max_steps}: "
                      f"loss={avg_loss:.4f}, "
                      f"lr={model_engine.get_lr()[0]:.2e}, "
                      f"samples/s={samples_per_sec:.1f}")
                
                total_loss = 0.0
            
            # Checkpointing
            if global_step % 500 == 0:
                save_checkpoint(model_engine, config, global_step)
    
    if local_rank == 0:
        print(f"\nTraining completed in {time.time() - start_time:.1f}s")


def save_checkpoint(model_engine: 'DeepSpeedEngine', config: TrainingConfig, step: int):
    """Save DeepSpeed checkpoint."""
    checkpoint_dir = os.path.join(config.checkpoint_dir, f"step_{step}")
    model_engine.save_checkpoint(checkpoint_dir)
    
    if model_engine.local_rank == 0:
        print(f"Saved checkpoint to {checkpoint_dir}")


def load_checkpoint(model_engine: 'DeepSpeedEngine', checkpoint_dir: str):
    """Load DeepSpeed checkpoint."""
    _, client_state = model_engine.load_checkpoint(checkpoint_dir)
    return client_state


# ============================================================================
# Main
# ============================================================================

def main():
    if not DEEPSPEED_AVAILABLE:
        print("DeepSpeed not available. Please install with: pip install deepspeed")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="DeepSpeed Training")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed_config", type=str, required=True)
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=1000)
    
    # Parse args including DeepSpeed args
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.hidden_size // 64,
        intermediate_size=args.hidden_size * 4,
        batch_size_per_gpu=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps,
    )
    
    # Load DeepSpeed config
    ds_config = create_deepspeed_config(config, args.deepspeed_config)
    
    # Initialize distributed
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if local_rank == 0:
        print("=" * 60)
        print("DEEPSPEED TRAINING")
        print("=" * 60)
        print(f"DeepSpeed config: {args.deepspeed_config}")
        print(f"ZeRO Stage: {ds_config.get('zero_optimization', {}).get('stage', 0)}")
        print(f"Model: {config.num_layers} layers, hidden={config.hidden_size}")
    
    # Create model
    model = GPTModel(config)
    
    if local_rank == 0:
        params_info = get_model_params_info(model)
        print(f"Parameters: {params_info['total_params_b']:.2f}B")
    
    # Create dataset
    dataset = SyntheticDataset(
        config.vocab_size,
        config.seq_length,
        config.num_samples,
    )
    
    # Initialize DeepSpeed
    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        config=ds_config,
    )
    
    if local_rank == 0:
        print(f"DeepSpeed engine initialized")
        print(f"World size: {model_engine.world_size}")
        print(f"Global batch size: {model_engine.train_batch_size()}")
        
        # Memory info
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        print("=" * 60)
    
    # Train
    train(model_engine, train_dataloader, config, local_rank)
    
    # Final save
    if local_rank == 0:
        save_checkpoint(model_engine, config, config.max_steps)
        print("Training complete!")


if __name__ == "__main__":
    main()
