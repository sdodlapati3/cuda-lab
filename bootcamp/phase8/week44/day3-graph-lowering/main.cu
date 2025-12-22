/**
 * Week 44, Day 3: Graph Lowering
 */
#include <cstdio>

int main() {
    printf("Week 44 Day 3: Graph Lowering Process\n\n");
    
    printf("From Python to CUDA:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ Step 1: TorchDynamo captures Python bytecode                      ║\n");
    printf("║ Step 2: Convert to FX Graph (torch.fx)                            ║\n");
    printf("║ Step 3: Decompose to ATen primitives                              ║\n");
    printf("║ Step 4: AOTAutograd traces forward + backward                     ║\n");
    printf("║ Step 5: Inductor schedules and generates code                     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("FX Graph Example:\n");
    printf("```python\n");
    printf("def fn(x, y):\n");
    printf("    return (x + y).relu()\n");
    printf("\n");
    printf("# FX Graph:\n");
    printf("# graph():\n");
    printf("#     %%x : [num_users=1] = placeholder[target=x]\n");
    printf("#     %%y : [num_users=1] = placeholder[target=y]\n");
    printf("#     %%add : [num_users=1] = call_function[target=torch.add](args=(%%x, %%y))\n");
    printf("#     %%relu : [num_users=1] = call_method[target=relu](args=(%%add,))\n");
    printf("#     return relu\n");
    printf("```\n\n");
    
    printf("Decomposition to Primitives:\n");
    printf("```python\n");
    printf("# High-level op:\n");
    printf("torch.nn.functional.layer_norm(x, [d], gamma, beta)\n");
    printf("\n");
    printf("# Decomposes to:\n");
    printf("mean = x.mean(dim=-1, keepdim=True)\n");
    printf("var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)\n");
    printf("rstd = 1 / torch.sqrt(var + eps)\n");
    printf("out = (x - mean) * rstd * gamma + beta\n");
    printf("```\n\n");
    
    printf("Why Decomposition Matters:\n");
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ • Simpler ops are easier to fuse                                    │\n");
    printf("│ • Enables more optimization opportunities                           │\n");
    printf("│ • Custom ops can participate in fusion                              │\n");
    printf("└─────────────────────────────────────────────────────────────────────┘\n");
    
    return 0;
}
