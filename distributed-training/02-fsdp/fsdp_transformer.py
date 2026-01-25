#!/usr/bin/env python3
"""
fsdp_transformer.py - Production FSDP training for large transformer models

Demonstrates:
- Efficient FSDP wrapping for transformers
- Activation checkpointing for memory efficiency
- Advanced mixed precision strategies
- Integration with HuggingFace models
- Multi-node training setup

Usage:
    # Single node, 4 GPUs
    torchrun --nproc_per_node=4 fsdp_transformer.py

    # Multi-node
    torchrun --nnodes=2 --node_rank=0 --master_addr=node0 \
             --master_port=29500 --nproc_per_node=4 fsdp_transformer.py

Author: CUDA Lab
"""

import os
import sys
import time
import argparse
import functools
import logging
from typing import Optional, Dict, Any, Set, Type
from dataclasses import dataclass, field
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.distributed.fsdp import StateDictType, FullStateDictConfig, ShardedStateDictConfig
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)


# Setup logging
def setup_logging(rank: int) -> logging.Logger:
    """Setup logging for distributed training."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(handler)
    
    return logger


@dataclass
class FSDPConfig:
    """Configuration for FSDP training."""
    # Model config
    model_name: str = "transformer"
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    
    # FSDP config
    sharding_strategy: str = "full_shard"
    use_mixed_precision: bool = True
    precision: str = "bf16"  # bf16 or fp16
    cpu_offload: bool = False
    use_activation_checkpointing: bool = True
    
    # Training config
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    num_epochs: int = 3
    
    # Performance
    use_compile: bool = False
    forward_prefetch: bool = True
    limit_all_gathers: bool = True


class RMSNorm(nn.Module):
    """RMSNorm for modern transformers."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos/sin
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x, seq_len: int):
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x):
    """Rotate half the hidden dims."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head attention with RoPE."""
    
    def __init__(self, config: FSDPConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim)
    
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(x, seq_len)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled dot-product attention (uses FlashAttention when available)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=True,
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class MLP(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    
    def __init__(self, config: FSDPConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerLayer(nn.Module):
    """Single transformer layer (for FSDP wrapping)."""
    
    def __init__(self, config: FSDPConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.attention = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.mlp = MLP(config)
    
    def forward(self, x, attention_mask=None):
        # Pre-norm architecture
        h = x + self.attention(self.input_layernorm(x), attention_mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class LLaMaModel(nn.Module):
    """LLaMA-style transformer for FSDP training."""
    
    def __init__(self, config: FSDPConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
        
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


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing."""
    
    def __init__(self, vocab_size: int, seq_len: int = 512, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
        # Pre-generate data for consistency
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        input_ids = self.data[idx]
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def get_fsdp_wrapped_model(model: nn.Module, config: FSDPConfig, rank: int) -> FSDP:
    """Wrap model with FSDP and apply activation checkpointing."""
    
    # Sharding strategy
    strategy_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }
    
    # Mixed precision
    mp_policy = None
    if config.use_mixed_precision:
        if config.precision == "bf16":
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        else:  # fp16
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
    
    # Auto wrap policy - wrap each TransformerLayer
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerLayer},
    )
    
    # FSDP config
    fsdp_kwargs = {
        "sharding_strategy": strategy_map[config.sharding_strategy],
        "mixed_precision": mp_policy,
        "auto_wrap_policy": auto_wrap_policy,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "forward_prefetch": config.forward_prefetch,
        "limit_all_gathers": config.limit_all_gathers,
        "use_orig_params": True,
        "device_id": torch.cuda.current_device(),
    }
    
    if config.cpu_offload:
        fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)
    
    # Wrap with FSDP
    model = FSDP(model, **fsdp_kwargs)
    
    # Apply activation checkpointing
    if config.use_activation_checkpointing:
        check_fn = lambda submodule: isinstance(submodule, TransformerLayer)
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=check_fn,
        )
        if rank == 0:
            print("Applied activation checkpointing to TransformerLayer modules")
    
    return model


def get_optimizer(model: nn.Module, config: FSDPConfig) -> optim.Optimizer:
    """Get optimizer with proper weight decay."""
    # Separate weight decay and no weight decay params
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "layernorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    return optim.AdamW(optimizer_groups, lr=config.learning_rate)


def get_scheduler(optimizer: optim.Optimizer, config: FSDPConfig, num_training_steps: int):
    """Get learning rate scheduler with warmup."""
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step):
        if current_step < config.warmup_steps:
            return float(current_step) / float(max(1, config.warmup_steps))
        return max(
            0.1,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - config.warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda)


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: optim.Optimizer,
    scheduler: Any,
    config: FSDPConfig,
    step: int,
    accumulation_step: int,
) -> float:
    """Execute single training step with gradient accumulation."""
    
    # Move batch to device
    input_ids = batch["input_ids"].cuda()
    labels = batch["labels"].cuda()
    
    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs["loss"] / config.gradient_accumulation_steps
    
    # Backward pass
    loss.backward()
    
    # Step optimizer after accumulation
    if (accumulation_step + 1) % config.gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return loss.item() * config.gradient_accumulation_steps


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Any,
    config: FSDPConfig,
    epoch: int,
    rank: int,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_steps = 0
    epoch_start = time.time()
    
    for step, batch in enumerate(dataloader):
        step_loss = train_step(
            model, batch, optimizer, scheduler, config, step,
            accumulation_step=step % config.gradient_accumulation_steps
        )
        
        total_loss += step_loss
        num_steps += 1
        
        # Logging
        if rank == 0 and step % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info(f"Epoch {epoch} Step {step}: loss={step_loss:.4f}, lr={lr:.2e}")
    
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / num_steps
    
    # Gather metrics across ranks
    loss_tensor = torch.tensor([avg_loss]).cuda()
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    
    return {
        "loss": loss_tensor.item(),
        "epoch_time": epoch_time,
        "samples_per_second": len(dataloader.dataset) / epoch_time,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    epoch: int,
    config: FSDPConfig,
    rank: int,
    checkpoint_dir: str = "checkpoints",
):
    """Save FSDP checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Use sharded state dict for distributed checkpoint
    with FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
    ):
        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }
        
        # Save per-rank checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_rank{rank}.pt")
        torch.save(state_dict, checkpoint_path)
    
    dist.barrier()
    
    if rank == 0:
        print(f"Saved checkpoint to {checkpoint_dir}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="FSDP Transformer Training")
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sharding", type=str, default="full_shard")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    
    # Setup logging
    logger = setup_logging(rank)
    
    # Create config
    config = FSDPConfig(
        num_hidden_layers=args.num_layers,
        hidden_size=args.hidden_size,
        intermediate_size=args.hidden_size * 4,
        num_attention_heads=args.hidden_size // 64,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        sharding_strategy=args.sharding,
        precision=args.precision,
        cpu_offload=args.cpu_offload,
        use_activation_checkpointing=not args.no_activation_checkpointing,
        use_compile=args.compile,
    )
    
    if rank == 0:
        logger.info("=" * 60)
        logger.info("FSDP Transformer Training")
        logger.info("=" * 60)
        logger.info(f"World size: {world_size}")
        logger.info(f"Model: {config.num_hidden_layers} layers, hidden={config.hidden_size}")
        logger.info(f"Sharding: {config.sharding_strategy}, Precision: {config.precision}")
        logger.info(f"Batch size: {config.batch_size}, Grad accum: {config.gradient_accumulation_steps}")
        logger.info(f"Activation checkpointing: {config.use_activation_checkpointing}")
        logger.info("=" * 60)
    
    # Create model
    model = LLaMaModel(config)
    num_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        logger.info(f"Model parameters: {num_params / 1e9:.2f}B")
    
    # Wrap with FSDP
    model = get_fsdp_wrapped_model(model, config, rank)
    
    # Optional: torch.compile
    if config.use_compile:
        if rank == 0:
            logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Create dataset and dataloader
    dataset = SyntheticDataset(config.vocab_size, seq_len=512, num_samples=10000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    
    # Optimizer and scheduler
    optimizer = get_optimizer(model, config)
    num_training_steps = len(dataloader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = get_scheduler(optimizer, config, num_training_steps)
    
    # Training loop
    if rank == 0:
        logger.info("Starting training...")
    
    for epoch in range(config.num_epochs):
        sampler.set_epoch(epoch)
        
        metrics = train_epoch(
            model, dataloader, optimizer, scheduler, config, epoch, rank, logger
        )
        
        if rank == 0:
            logger.info(f"Epoch {epoch + 1}: loss={metrics['loss']:.4f}, "
                       f"time={metrics['epoch_time']:.1f}s, "
                       f"throughput={metrics['samples_per_second']:.0f} samples/s")
            
            # Memory stats
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, config, rank)
    
    if rank == 0:
        logger.info("Training completed!")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
