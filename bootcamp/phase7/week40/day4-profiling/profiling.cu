/**
 * Week 40, Day 4: Performance Profiling Guide
 */
#include <cstdio>

int main() {
    printf("Week 40 Day 4: Profiling DL Kernels\n\n");
    
    printf("NSight Compute Metrics for DL Kernels:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ Memory-Bound Kernels (LayerNorm, Softmax, Fused Ops):             ║\n");
    printf("║   • DRAM Throughput: Should be near peak (~1.5 TB/s on A100)      ║\n");
    printf("║   • L2 Hit Rate: Higher is better for reused data                 ║\n");
    printf("║   • Memory Efficiency: Coalescing, bank conflicts                 ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║ Compute-Bound Kernels (GEMM, Attention Core):                     ║\n");
    printf("║   • SM Efficiency: Occupancy and warp scheduling                  ║\n");
    printf("║   • Tensor Core Utilization: Should be >80%% for GEMM             ║\n");
    printf("║   • Compute Throughput: FLOPS vs theoretical peak                 ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Profiling Commands:\n");
    printf("```bash\n");
    printf("# Quick overview\n");
    printf("ncu --target-processes all ./my_kernel\n");
    printf("\n");
    printf("# Memory analysis\n");
    printf("ncu --set memory --target-processes all ./my_kernel\n");
    printf("\n");
    printf("# Roofline analysis\n");
    printf("ncu --set roofline --target-processes all ./my_kernel\n");
    printf("\n");
    printf("# Full analysis (slower)\n");
    printf("ncu --set full --target-processes all -o profile ./my_kernel\n");
    printf("```\n\n");
    
    printf("Key Bottlenecks to Look For:\n");
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Bottleneck           │ Symptom                │ Solution            │\n");
    printf("├─────────────────────────────────────────────────────────────────────┤\n");
    printf("│ Memory bandwidth     │ Low SM efficiency,     │ Reduce traffic,     │\n");
    printf("│                      │ high DRAM throughput   │ fuse kernels        │\n");
    printf("│ Low occupancy        │ Warp stalls, low SM    │ Reduce registers,   │\n");
    printf("│                      │ efficiency             │ adjust block size   │\n");
    printf("│ Bank conflicts       │ High shared memory     │ Pad shared arrays   │\n");
    printf("│                      │ replays                │                     │\n");
    printf("│ Divergence           │ High instruction       │ Restructure control │\n");
    printf("│                      │ replays                │ flow                │\n");
    printf("└─────────────────────────────────────────────────────────────────────┘\n\n");
    
    printf("PyTorch Profiler Integration:\n");
    printf("```python\n");
    printf("with torch.profiler.profile(\n");
    printf("    activities=[torch.profiler.ProfilerActivity.CUDA],\n");
    printf("    with_stack=True,\n");
    printf("    record_shapes=True\n");
    printf(") as prof:\n");
    printf("    model(input)\n");
    printf("print(prof.key_averages().table(sort_by='cuda_time_total'))\n");
    printf("```\n");
    
    return 0;
}
