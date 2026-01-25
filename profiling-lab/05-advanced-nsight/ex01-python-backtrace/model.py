"""
Model with intentional performance bottleneck for profiling exercise.

One of these layers has a performance issue. Use Python backtraces to find it!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Standard positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EfficientAttention(nn.Module):
    """Efficient multi-head attention using PyTorch's SDPA."""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
    
    def forward(self, x):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        
        # Efficient attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Standard feedforward network."""
    
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Standard transformer block."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.attention = EfficientAttention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class HeavyLayer(nn.Module):
    """
    INTENTIONAL BOTTLENECK - This layer has performance issues!
    
    Can you identify why this is slow using profiling?
    
    Issues:
    1. Unnecessary CPU-GPU synchronization
    2. Inefficient looped computation
    3. Non-fused operations
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(d_model))
        
        # Unnecessary extra matrices for "processing"
        self.aux_weight1 = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.aux_weight2 = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
    
    def forward(self, x):
        """
        Intentionally inefficient forward pass.
        A good profiler should show this as the bottleneck.
        """
        B, T, C = x.shape
        
        # Issue 1: Loop over sequence (should be batched!)
        outputs = []
        for t in range(T):
            token = x[:, t, :]  # [B, C]
            
            # Issue 2: Multiple separate matmuls (should be fused)
            h1 = torch.matmul(token, self.weight)
            h2 = torch.matmul(h1, self.aux_weight1)
            h3 = torch.matmul(h2, self.aux_weight2)
            
            # Issue 3: Element-wise ops not fused with matmul
            h3 = h3 + self.bias
            h3 = torch.tanh(h3)
            
            outputs.append(h3)
        
        # Issue 4: Stack creates new tensor (memory inefficient)
        result = torch.stack(outputs, dim=1)
        
        return result


class TransformerWithHeavyLayer(nn.Module):
    """
    Transformer model with an intentional performance bottleneck.
    
    Architecture:
    - Embedding + Positional Encoding
    - N Transformer Blocks (efficient)
    - HeavyLayer (BOTTLENECK - find this!)
    - Output projection
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_classes: int = 10
    ):
        super().__init__()
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks (these are efficient)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # THE BOTTLENECK - HeavyLayer
        self.heavy_layer = HeavyLayer(d_model)
        
        # Output
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Heavy layer (BOTTLENECK)
        x = self.heavy_layer(x)
        
        # Classification (mean pooling)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        
        return x


class EfficientModel(nn.Module):
    """
    Fixed version without HeavyLayer bottleneck.
    Use this to compare profiling results.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_classes: int = 10
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Efficient replacement for HeavyLayer
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.projection(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        
        return x
