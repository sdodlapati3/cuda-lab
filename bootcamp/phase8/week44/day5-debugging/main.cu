/**
 * Week 44, Day 5: Debugging torch.compile
 */
#include <cstdio>

int main() {
    printf("Week 44 Day 5: Debugging torch.compile\n\n");
    
    printf("Common Issues:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ 1. Graph Breaks: Parts of code can't be captured                  ║\n");
    printf("║ 2. Recompilation: Same code compiled multiple times               ║\n");
    printf("║ 3. Slowdowns: Compiled code slower than eager                     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Finding Graph Breaks:\n");
    printf("```python\n");
    printf("import torch._dynamo as dynamo\n");
    printf("\n");
    printf("# Explain what's happening\n");
    printf("dynamo.explain(my_function)(input)\n");
    printf("\n");
    printf("# Or with environment variable\n");
    printf("# TORCH_LOGS='graph_breaks' python script.py\n");
    printf("\n");
    printf("# Common graph break causes:\n");
    printf("# - print() statements\n");
    printf("# - Python built-ins on tensors (list(), len())\n");
    printf("# - Data-dependent control flow\n");
    printf("# - Unsupported ops\n");
    printf("```\n\n");
    
    printf("Avoiding Recompilation:\n");
    printf("```python\n");
    printf("# Bad: Recompiles for each batch size\n");
    printf("for batch in dataloader:  # Different sizes\n");
    printf("    output = compiled_model(batch)\n");
    printf("\n");
    printf("# Good: Use dynamic shapes\n");
    printf("compiled_model = torch.compile(model, dynamic=True)\n");
    printf("\n");
    printf("# Or mark specific dims as dynamic\n");
    printf("torch._dynamo.mark_dynamic(tensor, dim=0)  # Batch dim\n");
    printf("```\n\n");
    
    printf("Profiling:\n");
    printf("```python\n");
    printf("# Compare eager vs compiled\n");
    printf("import torch.utils.benchmark as benchmark\n");
    printf("\n");
    printf("eager_fn = my_function\n");
    printf("compiled_fn = torch.compile(my_function)\n");
    printf("\n");
    printf("# Warmup\n");
    printf("for _ in range(3):\n");
    printf("    compiled_fn(input)\n");
    printf("\n");
    printf("t_eager = benchmark.Timer(stmt='fn(x)', globals={'fn': eager_fn, 'x': input})\n");
    printf("t_compiled = benchmark.Timer(stmt='fn(x)', globals={'fn': compiled_fn, 'x': input})\n");
    printf("\n");
    printf("print(f'Eager: {t_eager.timeit(100).mean * 1000:.2f} ms')\n");
    printf("print(f'Compiled: {t_compiled.timeit(100).mean * 1000:.2f} ms')\n");
    printf("```\n");
    
    return 0;
}
