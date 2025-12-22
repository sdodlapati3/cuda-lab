/**
 * Week 42, Day 4: Custom Autograd Function
 */
#include <cstdio>

int main() {
    printf("Week 42 Day 4: Complete Custom Autograd Function\n\n");
    
    printf("Full Example: Fused LayerNorm\n");
    printf("══════════════════════════════════════════════════════════════════════\n\n");
    
    printf("```python\n");
    printf("import torch\n");
    printf("import my_layernorm_cuda\n");
    printf("\n");
    printf("class FusedLayerNorm(torch.autograd.Function):\n");
    printf("    @staticmethod\n");
    printf("    def forward(ctx, x, gamma, beta, eps=1e-5):\n");
    printf("        # Call CUDA forward kernel\n");
    printf("        y, mean, rstd = my_layernorm_cuda.forward(x, gamma, beta, eps)\n");
    printf("        \n");
    printf("        # Save for backward\n");
    printf("        ctx.save_for_backward(x, gamma, mean, rstd)\n");
    printf("        ctx.eps = eps\n");
    printf("        \n");
    printf("        return y\n");
    printf("    \n");
    printf("    @staticmethod\n");
    printf("    def backward(ctx, grad_output):\n");
    printf("        x, gamma, mean, rstd = ctx.saved_tensors\n");
    printf("        \n");
    printf("        # Call CUDA backward kernel\n");
    printf("        grad_x, grad_gamma, grad_beta = my_layernorm_cuda.backward(\n");
    printf("            grad_output, x, gamma, mean, rstd\n");
    printf("        )\n");
    printf("        \n");
    printf("        # Return grads in same order as forward inputs\n");
    printf("        # (x, gamma, beta, eps) -> (grad_x, grad_gamma, grad_beta, None)\n");
    printf("        return grad_x, grad_gamma, grad_beta, None\n");
    printf("\n");
    printf("# nn.Module wrapper\n");
    printf("class FusedLayerNormModule(torch.nn.Module):\n");
    printf("    def __init__(self, dim, eps=1e-5):\n");
    printf("        super().__init__()\n");
    printf("        self.gamma = torch.nn.Parameter(torch.ones(dim))\n");
    printf("        self.beta = torch.nn.Parameter(torch.zeros(dim))\n");
    printf("        self.eps = eps\n");
    printf("    \n");
    printf("    def forward(self, x):\n");
    printf("        return FusedLayerNorm.apply(x, self.gamma, self.beta, self.eps)\n");
    printf("```\n");
    
    return 0;
}
