/**
 * Week 43, Day 3: Softmax in Triton
 */
#include <cstdio>

int main() {
    printf("Week 43 Day 3: Softmax in Triton\n\n");
    
    printf("Triton Softmax:\n");
    printf("```python\n");
    printf("@triton.jit\n");
    printf("def softmax_kernel(input_ptr, output_ptr, n_cols,\n");
    printf("                   BLOCK_SIZE: tl.constexpr):\n");
    printf("    # Each program handles one row\n");
    printf("    row_idx = tl.program_id(0)\n");
    printf("    col_offsets = tl.arange(0, BLOCK_SIZE)\n");
    printf("    mask = col_offsets < n_cols\n");
    printf("    \n");
    printf("    # Load row\n");
    printf("    row_ptr = input_ptr + row_idx * n_cols\n");
    printf("    row = tl.load(row_ptr + col_offsets, mask=mask, other=-float('inf'))\n");
    printf("    \n");
    printf("    # Safe softmax\n");
    printf("    row_max = tl.max(row, axis=0)\n");
    printf("    numerator = tl.exp(row - row_max)\n");
    printf("    denominator = tl.sum(numerator, axis=0)\n");
    printf("    softmax_out = numerator / denominator\n");
    printf("    \n");
    printf("    # Store\n");
    printf("    out_ptr = output_ptr + row_idx * n_cols\n");
    printf("    tl.store(out_ptr + col_offsets, softmax_out, mask=mask)\n");
    printf("\n");
    printf("def softmax(x):\n");
    printf("    n_rows, n_cols = x.shape\n");
    printf("    BLOCK_SIZE = triton.next_power_of_2(n_cols)\n");
    printf("    output = torch.empty_like(x)\n");
    printf("    softmax_kernel[(n_rows,)](x, output, n_cols, BLOCK_SIZE=BLOCK_SIZE)\n");
    printf("    return output\n");
    printf("```\n\n");
    
    printf("Comparison with CUDA:\n");
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Feature           │ CUDA                 │ Triton               │\n");
    printf("├─────────────────────────────────────────────────────────────────────┤\n");
    printf("│ Lines of code     │ ~100                 │ ~20                  │\n");
    printf("│ Shared memory     │ Manual               │ Automatic            │\n");
    printf("│ Reductions        │ Manual __shfl        │ tl.max, tl.sum       │\n");
    printf("│ Performance       │ Peak possible        │ ~90%% of peak         │\n");
    printf("│ Development time  │ Hours                │ Minutes              │\n");
    printf("└─────────────────────────────────────────────────────────────────────┘\n");
    
    return 0;
}
