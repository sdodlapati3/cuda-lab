/**
 * Week 46, Day 1: NCCL Debugging
 */
#include <cstdio>

int main() {
    printf("Week 46 Day 1: Debugging NCCL\n\n");
    
    printf("Debug Environment Variables:\n");
    printf("```bash\n");
    printf("# Verbosity levels\n");
    printf("export NCCL_DEBUG=WARN   # Warnings only\n");
    printf("export NCCL_DEBUG=INFO   # General info\n");
    printf("export NCCL_DEBUG=TRACE  # Detailed tracing (verbose!)\n");
    printf("\n");
    printf("# Subsystem filtering\n");
    printf("export NCCL_DEBUG_SUBSYS=INIT,COLL  # Only init and collectives\n");
    printf("# Options: INIT, COLL, P2P, SHM, NET, GRAPH, TUNING, ENV, ALL\n");
    printf("\n");
    printf("# Log to file\n");
    printf("export NCCL_DEBUG_FILE=/tmp/nccl_%%h_%%p.log\n");
    printf("# %%h = hostname, %%p = pid\n");
    printf("```\n\n");
    
    printf("Common Issues:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ Hang: All ranks must call collective together                     ║\n");
    printf("║ CUDA error: Check CUDA_VISIBLE_DEVICES matches rank               ║\n");
    printf("║ Timeout: Increase NCCL_TIMEOUT (seconds)                          ║\n");
    printf("║ OOM: Reduce NCCL_BUFFSIZE or batch size                           ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("PyTorch Debug:\n");
    printf("```python\n");
    printf("# Enable NCCL debug in PyTorch\n");
    printf("import torch.distributed as dist\n");
    printf("dist.init_process_group(..., timeout=datetime.timedelta(minutes=10))\n");
    printf("\n");
    printf("# Check if all ranks reached a point\n");
    printf("dist.barrier()  # Will hang if any rank is missing\n");
    printf("```\n");
    
    return 0;
}
