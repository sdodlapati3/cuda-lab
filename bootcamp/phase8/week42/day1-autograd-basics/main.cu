/**
 * Week 42, Day 1: Autograd Basics
 */
#include <cstdio>

int main() {
    printf("Week 42 Day 1: PyTorch Autograd Basics\n\n");
    
    printf("How Autograd Works:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ 1. Forward pass builds computation graph                          ║\n");
    printf("║ 2. Each tensor tracks its grad_fn (how it was created)            ║\n");
    printf("║ 3. Backward pass traverses graph, computing gradients             ║\n");
    printf("║ 4. Chain rule: dL/dx = dL/dy × dy/dx                              ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("For Custom CUDA Ops:\n");
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ You must implement both:                                            │\n");
    printf("│   • forward(): compute output from input                            │\n");
    printf("│   • backward(): compute grad_input from grad_output                 │\n");
    printf("└─────────────────────────────────────────────────────────────────────┘\n\n");
    
    printf("torch.autograd.Function:\n");
    printf("```python\n");
    printf("class MyReLU(torch.autograd.Function):\n");
    printf("    @staticmethod\n");
    printf("    def forward(ctx, input):\n");
    printf("        ctx.save_for_backward(input)\n");
    printf("        return my_cuda_ext.relu_forward(input)\n");
    printf("    \n");
    printf("    @staticmethod\n");
    printf("    def backward(ctx, grad_output):\n");
    printf("        input, = ctx.saved_tensors\n");
    printf("        return my_cuda_ext.relu_backward(grad_output, input)\n");
    printf("\n");
    printf("# Usage\n");
    printf("my_relu = MyReLU.apply\n");
    printf("output = my_relu(input)\n");
    printf("```\n");
    
    return 0;
}
