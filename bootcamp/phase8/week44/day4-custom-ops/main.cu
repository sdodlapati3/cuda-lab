/**
 * Week 44, Day 4: Custom Ops with torch.compile
 */
#include <cstdio>

int main() {
    printf("Week 44 Day 4: Custom Ops with torch.compile\n\n");
    
    printf("Making Custom Ops Compile-Friendly:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ Your CUDA extension needs to tell the compiler:                   ║\n");
    printf("║ • What shape/dtype the output will have                           ║\n");
    printf("║ • Whether it's safe to fuse                                       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Registering with torch.library:\n");
    printf("```python\n");
    printf("import torch\n");
    printf("from torch.library import Library, impl\n");
    printf("\n");
    printf("# Define the op schema\n");
    printf("my_lib = Library('my_ops', 'DEF')\n");
    printf("my_lib.define('my_relu(Tensor x) -> Tensor')\n");
    printf("\n");
    printf("# CUDA implementation\n");
    printf("@impl(my_lib, 'my_relu', 'CUDA')\n");
    printf("def my_relu_cuda(x):\n");
    printf("    return my_cuda_ext.relu(x)\n");
    printf("\n");
    printf("# Meta implementation (for shape inference)\n");
    printf("@impl(my_lib, 'my_relu', 'Meta')\n");
    printf("def my_relu_meta(x):\n");
    printf("    return torch.empty_like(x)\n");
    printf("\n");
    printf("# Now works with torch.compile!\n");
    printf("@torch.compile\n");
    printf("def use_custom_op(x):\n");
    printf("    return torch.ops.my_ops.my_relu(x)\n");
    printf("```\n\n");
    
    printf("Backward Support:\n");
    printf("```python\n");
    printf("# Register backward\n");
    printf("my_lib.define('my_relu_backward(Tensor grad, Tensor x) -> Tensor')\n");
    printf("\n");
    printf("@impl(my_lib, 'my_relu_backward', 'CUDA')\n");
    printf("def my_relu_backward_cuda(grad, x):\n");
    printf("    return my_cuda_ext.relu_backward(grad, x)\n");
    printf("\n");
    printf("# Register autograd formula\n");
    printf("def setup_context(ctx, inputs, output):\n");
    printf("    x, = inputs\n");
    printf("    ctx.save_for_backward(x)\n");
    printf("\n");
    printf("def backward(ctx, grad):\n");
    printf("    x, = ctx.saved_tensors\n");
    printf("    return torch.ops.my_ops.my_relu_backward(grad, x)\n");
    printf("\n");
    printf("torch.library.register_autograd('my_ops::my_relu', backward, setup_context)\n");
    printf("```\n");
    
    return 0;
}
