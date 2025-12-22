/**
 * Week 48, Day 1: Gradient Accumulation
 */
#include <cstdio>

int main() {
    printf("Week 48 Day 1: Gradient Accumulation\n\n");
    
    printf("Why Gradient Accumulation?\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ • Simulate larger batch sizes without more memory                 ║\n");
    printf("║ • Effective batch = micro_batch × accum_steps × num_gpus          ║\n");
    printf("║ • Example: 8 × 4 × 8 = 256 effective batch size                   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Implementation:\n");
    printf("```python\n");
    printf("accumulation_steps = 4\n");
    printf("\n");
    printf("for i, batch in enumerate(dataloader):\n");
    printf("    # Forward + backward (gradients accumulate)\n");
    printf("    loss = model(batch).sum() / accumulation_steps\n");
    printf("    loss.backward()\n");
    printf("    \n");
    printf("    if (i + 1) %% accumulation_steps == 0:\n");
    printf("        optimizer.step()  # Update weights\n");
    printf("        optimizer.zero_grad()  # Clear accumulated gradients\n");
    printf("```\n\n");
    
    printf("With DDP:\n");
    printf("```python\n");
    printf("# Disable sync during accumulation for efficiency\n");
    printf("for i, batch in enumerate(dataloader):\n");
    printf("    # Only sync on last accumulation step\n");
    printf("    sync_grads = (i + 1) %% accumulation_steps == 0\n");
    printf("    \n");
    printf("    with model.no_sync() if not sync_grads else nullcontext():\n");
    printf("        loss = model(batch).sum() / accumulation_steps\n");
    printf("        loss.backward()\n");
    printf("    \n");
    printf("    if sync_grads:\n");
    printf("        optimizer.step()\n");
    printf("        optimizer.zero_grad()\n");
    printf("```\n");
    
    return 0;
}
