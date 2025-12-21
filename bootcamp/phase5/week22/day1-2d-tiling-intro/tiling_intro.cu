/**
 * Day 1: 2D Tiling Introduction
 * 
 * Demonstrates the theory behind 2D tiling:
 * - Data reuse calculations
 * - Memory traffic analysis
 * - Tile size trade-offs
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Calculate data reuse and memory traffic for different approaches
void analyzeDataReuse(int M, int N, int K) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║              2D Tiling Data Reuse Analysis                   ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Total output elements: %d\n", M * N);
    printf("Total FLOPs: %.2f GFLOP\n\n", 2.0 * M * N * K / 1e9);
    
    // Naive analysis
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("Naive GEMM (no tiling):\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    // Each output element reads K elements from A and K from B
    // Total reads: M × N × (K + K) = 2 × M × N × K
    long long naiveReads = 2LL * M * N * K;
    float naiveGB = naiveReads * sizeof(float) / 1e9;
    printf("  Each output reads: %d from A + %d from B = %d floats\n", K, K, 2*K);
    printf("  Total global reads: %lld (%.2f GB)\n", naiveReads, naiveGB);
    printf("  Arithmetic intensity: %.2f FLOP/byte\n", 
           2.0 * M * N * K / (naiveGB * 1e9));
    printf("  On A100 (2039 GB/s): %.2f ms (memory bound)\n\n", 
           naiveGB * 1000 / 2039);
    
    // Tiled analysis
    int tileSizes[] = {8, 16, 32, 64};
    
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("2D Tiled GEMM:\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    printf("┌──────────┬────────────┬──────────┬────────────┬────────────┐\n");
    printf("│ Tile     │ Shared Mem │ Reuse    │ Global GB  │ Est. Time  │\n");
    printf("│ Size     │ per Block  │ Factor   │ Reads      │ (A100)     │\n");
    printf("├──────────┼────────────┼──────────┼────────────┼────────────┤\n");
    
    for (int tile : tileSizes) {
        // Shared memory: 2 tiles of TILE × TILE
        int sharedMem = 2 * tile * tile * sizeof(float);
        
        // Data reuse factor = tile size
        int reuseFactor = tile;
        
        // Reduced reads: naive / reuse_factor
        long long tiledReads = naiveReads / reuseFactor;
        float tiledGB = tiledReads * sizeof(float) / 1e9;
        
        // Estimate time (memory bound)
        float estTime = tiledGB * 1000 / 2039;  // ms
        
        printf("│ %3d×%-3d  │ %6d KB  │   %3d×   │ %8.2f   │ %8.2f ms │\n",
               tile, tile, sharedMem / 1024, reuseFactor, tiledGB, estTime);
    }
    printf("└──────────┴────────────┴──────────┴────────────┴────────────┘\n\n");
    
    // Occupancy analysis
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("Occupancy Considerations (A100: 164KB shared/SM):\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    printf("┌──────────┬────────────┬────────────┬───────────────────────┐\n");
    printf("│ Tile     │ Shared/Blk │ Blocks/SM  │ Notes                 │\n");
    printf("├──────────┼────────────┼────────────┼───────────────────────┤\n");
    
    for (int tile : tileSizes) {
        int sharedMem = 2 * tile * tile * sizeof(float);
        int blocksPerSM = 164 * 1024 / sharedMem;
        if (blocksPerSM > 32) blocksPerSM = 32;  // Max blocks per SM
        
        const char* notes = "";
        if (tile == 8) notes = "High occupancy, low reuse";
        else if (tile == 16) notes = "Good balance";
        else if (tile == 32) notes = "Best for most cases";
        else if (tile == 64) notes = "Low occupancy, high reuse";
        
        printf("│ %3d×%-3d  │ %6d KB  │ %6d     │ %-21s │\n",
               tile, tile, sharedMem / 1024, blocksPerSM, notes);
    }
    printf("└──────────┴────────────┴────────────┴───────────────────────┘\n\n");
    
    // Key insight
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                      Key Insight                             ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ 2D tiling reduces memory traffic by TILE_SIZE factor         ║\n");
    printf("║ TILE=32 is often optimal: 32× fewer global reads,            ║\n");
    printf("║ reasonable shared memory usage, good occupancy.              ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
}

// Visualize how tiles move through the matrices
void visualizeTileMovement() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║              Tile Movement Visualization                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Computing C[blockRow][blockCol] tile:\n\n");
    
    printf("Iteration 0 (k=0):\n");
    printf("  A: Load tile A[blockRow, 0]     B: Load tile B[0, blockCol]\n");
    printf("  ┌─────────────────┐             ┌─────────────────┐\n");
    printf("  │ ████ │    │    │             │ ████ │    │    │\n");
    printf("  │ ████ │    │    │             ├──────┼────┼────┤\n");
    printf("  ├──────┼────┼────┤     ×       │      │    │    │\n");
    printf("  │      │    │    │             ├──────┼────┼────┤\n");
    printf("  └─────────────────┘             │      │    │    │\n");
    printf("                                  └─────────────────┘\n\n");
    
    printf("Iteration 1 (k=TILE_K):\n");
    printf("  A: Load tile A[blockRow, 1]     B: Load tile B[1, blockCol]\n");
    printf("  ┌─────────────────┐             ┌─────────────────┐\n");
    printf("  │      │ ████ │  │             │      │    │    │\n");
    printf("  │      │ ████ │  │             ├──────┼────┼────┤\n");
    printf("  ├──────┼──────┼──┤     ×       │ ████ │    │    │\n");
    printf("  │      │      │  │             ├──────┼────┼────┤\n");
    printf("  └─────────────────┘             │      │    │    │\n");
    printf("                                  └─────────────────┘\n\n");
    
    printf("... continue for K/TILE_K iterations ...\n\n");
    
    printf("After all iterations:\n");
    printf("  C: Write tile C[blockRow, blockCol]\n");
    printf("  ┌─────────────────┐\n");
    printf("  │ ████ │    │    │  ← Accumulated result from all k iterations\n");
    printf("  ├──────┼────┼────┤\n");
    printf("  │      │    │    │\n");
    printf("  └─────────────────┘\n");
}

int main() {
    printf("\n");
    
    // Analyze for typical GEMM size
    analyzeDataReuse(2048, 2048, 2048);
    
    // Visualize tile movement
    visualizeTileMovement();
    
    printf("\n");
    printf("Tomorrow: Implement basic tiled GEMM kernel!\n");
    printf("\n");
    
    return 0;
}
