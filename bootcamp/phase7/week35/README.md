# Week 35: Attention - QK^T and Masking

Building blocks for attention computation: Q×K^T scores and masking.

## Overview
- Days 1-2: QK^T computation (batched matmul)
- Days 3-4: Causal masking (autoregressive)
- Days 5-6: Combined attention scores

## Key Concepts
- Batched matrix multiplication for multi-head attention
- Causal mask: upper triangle = -inf
- Padding mask: variable sequence lengths
- Memory-efficient score computation
