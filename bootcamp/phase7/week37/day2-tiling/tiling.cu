/**
 * Week 37, Day 2: FlashAttention Tiling Strategy
 * 
 * Tile sizes determined by SRAM (shared memory) capacity.
 * Goal: Fit Q, K, V tiles and partial S in SRAM.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// A100 has 192KB shared memory per SM (configurable)
// Typical allocation: 48KB for attention tiles

void calculateTileSizes(int sram_size_kb, int d) {
    int sram_bytes = sram_size_kb * 1024;
    
    // FlashAttention tiling:
    // Need space for: Q_tile, K_tile, V_tile, S_tile, O_tile
    // 
    // Q_tile: [Br, d] - rows of queries
    // K_tile: [Bc, d] - rows of keys
    // V_tile: [Bc, d] - rows of values
    // S_tile: [Br, Bc] - partial attention scores
    // O_tile: [Br, d] - output accumulator
    //
    // Total: Br*d + 2*Bc*d + Br*Bc + Br*d = 2*Br*d + 2*Bc*d + Br*Bc
    
    printf("SRAM Budget: %d KB, Head dim: %d\n\n", sram_size_kb, d);
    
    // Try different tile sizes
    printf("Tile Size Analysis (float32):\n");
    printf("┌──────┬──────┬────────────┬───────────────────────────────────┐\n");
    printf("│  Br  │  Bc  │ SRAM (KB)  │ Configuration                     │\n");
    printf("├──────┼──────┼────────────┼───────────────────────────────────┤\n");
    
    int tile_sizes[] = {16, 32, 64, 128, 256};
    for (int br_idx = 0; br_idx < 5; br_idx++) {
        for (int bc_idx = 0; bc_idx < 5; bc_idx++) {
            int Br = tile_sizes[br_idx];
            int Bc = tile_sizes[bc_idx];
            
            // Q_tile + K_tile + V_tile + S_tile + O_tile + extras
            int q_tile = Br * d;
            int k_tile = Bc * d;
            int v_tile = Bc * d;
            int s_tile = Br * Bc;
            int o_tile = Br * d;
            int total_elements = q_tile + k_tile + v_tile + s_tile + o_tile;
            int total_bytes = total_elements * sizeof(float);
            float total_kb = total_bytes / 1024.0f;
            
            if (total_kb <= sram_size_kb && total_kb >= sram_size_kb * 0.5) {
                const char* status = total_kb <= sram_size_kb ? "✓ Fits" : "✗ Too big";
                printf("│ %4d │ %4d │ %8.1f   │ %s                          │\n", 
                       Br, Bc, total_kb, status);
            }
        }
    }
    printf("└──────┴──────┴────────────┴───────────────────────────────────┘\n\n");
}

__global__ void demonstrateTiling() {
    // Show the tiled loop structure
    // This is pseudocode in kernel form for illustration
    
    /*
    FlashAttention forward pass (pseudocode):
    
    // Outer loop: iterate over K, V blocks
    for (int j = 0; j < num_kv_blocks; j++) {
        // Load K_j, V_j tiles into SRAM
        load_kv_tile(j);
        
        // Inner loop: iterate over Q blocks  
        for (int i = 0; i < num_q_blocks; i++) {
            // Load Q_i tile into SRAM
            load_q_tile(i);
            
            // Compute S_ij = Q_i @ K_j^T (in SRAM)
            compute_qkt_tile();
            
            // Update running softmax: m, l
            update_online_softmax();
            
            // Update output: O_i = (old_O_i * rescale) + (P_ij @ V_j)
            update_output_tile();
        }
    }
    */
}

int main() {
    printf("Week 37 Day 2: FlashAttention Tiling\n\n");
    
    printf("Tiling Visualization:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║                                                                   ║\n");
    printf("║    Q [N×d]         K^T [d×N]         S [N×N]                      ║\n");
    printf("║   ┌─────────┐     ┌─────────┐     ┌───┬───┬───┬───┐              ║\n");
    printf("║   │  Q_0    │  ×  │K0│K1│K2│K3│ = │S00│S01│S02│S03│              ║\n");
    printf("║   ├─────────┤     └─────────┘     ├───┼───┼───┼───┤              ║\n");
    printf("║   │  Q_1    │                     │S10│S11│S12│S13│              ║\n");
    printf("║   ├─────────┤        Compute      ├───┼───┼───┼───┤              ║\n");
    printf("║   │  Q_2    │        tiles        │S20│S21│S22│S23│              ║\n");
    printf("║   ├─────────┤        one at       ├───┼───┼───┼───┤              ║\n");
    printf("║   │  Q_3    │        a time       │S30│S31│S32│S33│              ║\n");
    printf("║   └─────────┘                     └───┴───┴───┴───┘              ║\n");
    printf("║                                                                   ║\n");
    printf("║   Key: Never store full S matrix! Keep S_ij tile in SRAM only.   ║\n");
    printf("║                                                                   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    // Different SRAM budgets
    calculateTileSizes(48, 64);   // Conservative
    calculateTileSizes(96, 64);   // Moderate
    calculateTileSizes(48, 128);  // Larger head dim
    
    printf("FlashAttention Loop Structure:\n");
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ // Outer loop: K, V blocks (along sequence dim)                 │\n");
    printf("│ for j in range(N / Bc):                                         │\n");
    printf("│     K_j = K[j*Bc : (j+1)*Bc]  # Load from HBM → SRAM            │\n");
    printf("│     V_j = V[j*Bc : (j+1)*Bc]  # Load from HBM → SRAM            │\n");
    printf("│                                                                 │\n");
    printf("│     // Inner loop: Q blocks                                     │\n");
    printf("│     for i in range(N / Br):                                     │\n");
    printf("│         Q_i = Q[i*Br : (i+1)*Br]  # Load from HBM → SRAM        │\n");
    printf("│         O_i, m_i, l_i = load_state(i)                           │\n");
    printf("│                                                                 │\n");
    printf("│         S_ij = Q_i @ K_j.T  # [Br, Bc] in SRAM                  │\n");
    printf("│         m_new, l_new = online_softmax(S_ij, m_i, l_i)           │\n");
    printf("│         O_i = rescale(O_i) + softmax(S_ij) @ V_j                │\n");
    printf("│                                                                 │\n");
    printf("│         save_state(i, O_i, m_new, l_new)                        │\n");
    printf("└─────────────────────────────────────────────────────────────────┘\n\n");
    
    printf("Why This Loop Order?\n");
    printf("  • K, V are read once per outer loop (N/Bc times total)\n");
    printf("  • Q is read once per inner loop (N/Br × N/Bc times total)\n");
    printf("  • But since Q loops are parallelized across blocks...\n");
    printf("  • Each Q element is read O(N/Bc) times, not O(N²)!\n");
    
    return 0;
}
