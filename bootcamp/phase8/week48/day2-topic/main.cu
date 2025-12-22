/**
 * Week 48, Day 2: Mixed Precision Distributed
 */
#include <cstdio>

int main() {
    printf("Week 48 Day 2: Mixed Precision Distributed\n\n");
    
    printf("AMP + DDP:\n");
    printf("```python\n");
    printf("from torch.cuda.amp import autocast, GradScaler\n");
    printf("\n");
    printf("model = DDP(model, device_ids=[rank])\n");
    printf("scaler = GradScaler()\n");
    printf("\n");
    printf("for batch in dataloader:\n");
    printf("    optimizer.zero_grad()\n");
    printf("    \n");
    printf("    with autocast():\n");
    printf("        loss = model(batch).sum()\n");
    printf("    \n");
    printf("    scaler.scale(loss).backward()  # Grads synced via NCCL\n");
    printf("    scaler.step(optimizer)\n");
    printf("    scaler.update()\n");
    printf("```\n\n");
    
    printf("BF16 vs FP16:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ FP16:                                                             ║\n");
    printf("║   • 5 exp bits, 10 mantissa bits                                  ║\n");
    printf("║   • Needs loss scaling (can overflow/underflow)                   ║\n");
    printf("║   • Supported on all modern GPUs                                  ║\n");
    printf("║                                                                   ║\n");
    printf("║ BF16:                                                             ║\n");
    printf("║   • 8 exp bits, 7 mantissa bits (same range as FP32)              ║\n");
    printf("║   • No loss scaling needed!                                       ║\n");
    printf("║   • A100+, H100 only                                              ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("BF16 Training (simpler):\n");
    printf("```python\n");
    printf("with autocast(dtype=torch.bfloat16):  # No scaler needed!\n");
    printf("    loss = model(batch).sum()\n");
    printf("loss.backward()\n");
    printf("optimizer.step()\n");
    printf("```\n");
    
    return 0;
}
