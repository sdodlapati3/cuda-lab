/**
 * Week 42, Day 3: Gradient Checking
 */
#include <cstdio>

int main() {
    printf("Week 42 Day 3: Gradient Checking\n\n");
    
    printf("Why Gradient Check?\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ Backward kernels are error-prone. Numerical gradient check        ║\n");
    printf("║ verifies correctness by comparing:                                ║\n");
    printf("║   • Analytical gradient (your backward kernel)                    ║\n");
    printf("║   • Numerical gradient (finite differences)                       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("torch.autograd.gradcheck:\n");
    printf("```python\n");
    printf("from torch.autograd import gradcheck\n");
    printf("\n");
    printf("def test_gradients():\n");
    printf("    # Double precision for numerical stability\n");
    printf("    x = torch.randn(8, 64, device='cuda', dtype=torch.float64,\n");
    printf("                    requires_grad=True)\n");
    printf("    gamma = torch.randn(64, device='cuda', dtype=torch.float64,\n");
    printf("                        requires_grad=True)\n");
    printf("    beta = torch.randn(64, device='cuda', dtype=torch.float64,\n");
    printf("                       requires_grad=True)\n");
    printf("    \n");
    printf("    # gradcheck computes numerical gradient and compares\n");
    printf("    assert gradcheck(\n");
    printf("        MyLayerNorm.apply,\n");
    printf("        (x, gamma, beta),\n");
    printf("        eps=1e-6,\n");
    printf("        atol=1e-4,\n");
    printf("        rtol=1e-3\n");
    printf("    )\n");
    printf("```\n\n");
    
    printf("Common Pitfalls:\n");
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ • Use float64 for gradcheck (float32 too noisy)                     │\n");
    printf("│ • Small inputs (batch=8) for speed                                  │\n");
    printf("│ • Check all input tensors that require grad                         │\n");
    printf("│ • Adjust eps/atol/rtol if check fails                               │\n");
    printf("└─────────────────────────────────────────────────────────────────────┘\n");
    
    return 0;
}
