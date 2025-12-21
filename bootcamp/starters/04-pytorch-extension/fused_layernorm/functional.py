"""
Fused LayerNorm - Autograd Function

This module wraps the CUDA kernels with PyTorch autograd.
"""

import torch
from torch.autograd import Function

# Import the compiled CUDA extension
try:
    import fused_layernorm_cuda
except ImportError as e:
    raise ImportError(
        "Could not import fused_layernorm_cuda. "
        "Make sure to run: pip install -e . "
        f"Original error: {e}"
    )


class FusedLayerNormFunction(Function):
    """
    Custom autograd function for fused LayerNorm.
    
    Forward: Computes LayerNorm and saves tensors for backward
    Backward: Computes gradients for input, weight, and bias
    """
    
    @staticmethod
    def forward(ctx, input, weight, bias, eps):
        """
        Forward pass of fused LayerNorm.
        
        Args:
            ctx: Context object for saving tensors
            input: Input tensor (batch_size, hidden_size)
            weight: Learnable scale parameter (hidden_size,)
            bias: Learnable shift parameter (hidden_size,)
            eps: Small constant for numerical stability
        
        Returns:
            output: Normalized and transformed tensor
        """
        # Ensure contiguous
        input = input.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        
        # Call CUDA forward
        output, mean, rstd = fused_layernorm_cuda.forward(input, weight, bias, eps)
        
        # Save for backward
        ctx.save_for_backward(input, weight, mean, rstd)
        ctx.eps = eps
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of fused LayerNorm.
        
        Args:
            ctx: Context with saved tensors
            grad_output: Gradient from next layer
        
        Returns:
            Tuple of gradients: (grad_input, grad_weight, grad_bias, None)
            None for eps since it's not a tensor
        """
        input, weight, mean, rstd = ctx.saved_tensors
        
        # Ensure contiguous
        grad_output = grad_output.contiguous()
        
        # Call CUDA backward
        grad_input, grad_weight, grad_bias = fused_layernorm_cuda.backward(
            grad_output, input, weight, mean, rstd
        )
        
        # Return gradients (None for eps since it's not differentiable)
        return grad_input, grad_weight, grad_bias, None


def fused_layer_norm(input, weight, bias, eps=1e-5):
    """
    Applies fused Layer Normalization.
    
    This is a drop-in replacement for:
        torch.nn.functional.layer_norm(input, (hidden_size,), weight, bias, eps)
    
    Args:
        input: Input tensor of shape (batch_size, hidden_size)
        weight: Scale parameter of shape (hidden_size,)
        bias: Shift parameter of shape (hidden_size,)
        eps: Small constant for numerical stability (default: 1e-5)
    
    Returns:
        Normalized and transformed tensor of shape (batch_size, hidden_size)
    
    Example:
        >>> x = torch.randn(32, 1024, device='cuda', requires_grad=True)
        >>> w = torch.ones(1024, device='cuda', requires_grad=True)
        >>> b = torch.zeros(1024, device='cuda', requires_grad=True)
        >>> out = fused_layer_norm(x, w, b)
        >>> out.sum().backward()
    """
    return FusedLayerNormFunction.apply(input, weight, bias, eps)


class FusedLayerNorm(torch.nn.Module):
    """
    Fused Layer Normalization module.
    
    Drop-in replacement for torch.nn.LayerNorm.
    
    Args:
        hidden_size: Size of the last dimension
        eps: Small constant for numerical stability
    
    Example:
        >>> layer = FusedLayerNorm(1024).cuda()
        >>> x = torch.randn(32, 1024, device='cuda')
        >>> out = layer(x)
    """
    
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        # Learnable parameters (initialized like PyTorch's LayerNorm)
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, input):
        return fused_layer_norm(input, self.weight, self.bias, self.eps)
    
    def extra_repr(self):
        return f'{self.hidden_size}, eps={self.eps}'
