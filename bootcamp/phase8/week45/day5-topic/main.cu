/**
 * Week 45, Day 5: NCCL Performance
 */
#include <cstdio>

int main() {
    printf("Week 45 Day 5: NCCL Performance Tuning\n\n");
    
    printf("Key Performance Factors:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ 1. Interconnect: NVLink >> PCIe                                   ║\n");
    printf("║ 2. Topology: NCCL auto-detects optimal paths                      ║\n");
    printf("║ 3. Message size: Larger = better bandwidth utilization            ║\n");
    printf("║ 4. Overlap: Overlap communication with computation                ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Bandwidth Expectations (A100):\n");
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ NVLink (intra-node):  ~600 GB/s bidirectional                       │\n");
    printf("│ PCIe Gen4:            ~32 GB/s per direction                        │\n");
    printf("│ InfiniBand HDR:       ~200 Gb/s (~25 GB/s) inter-node               │\n");
    printf("└─────────────────────────────────────────────────────────────────────┘\n\n");
    
    printf("Environment Variables:\n");
    printf("```bash\n");
    printf("# Debug output\n");
    printf("export NCCL_DEBUG=INFO\n");
    printf("\n");
    printf("# Force specific algorithm\n");
    printf("export NCCL_ALGO=Ring  # or Tree\n");
    printf("\n");
    printf("# Tune buffer sizes\n");
    printf("export NCCL_BUFFSIZE=4194304  # 4MB\n");
    printf("\n");
    printf("# Multi-node networking\n");
    printf("export NCCL_IB_DISABLE=0  # Enable InfiniBand\n");
    printf("export NCCL_SOCKET_IFNAME=eth0\n");
    printf("```\n\n");
    
    printf("Benchmarking:\n");
    printf("```bash\n");
    printf("# nccl-tests (official benchmark)\n");
    printf("./build/all_reduce_perf -b 8 -e 128M -f 2 -g 4\n");
    printf("# -b: min bytes, -e: max bytes, -f: factor, -g: GPUs\n");
    printf("```\n");
    
    return 0;
}
