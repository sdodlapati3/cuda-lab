/**
 * Week 41, Day 5: Testing Extensions
 */
#include <cstdio>

int main() {
    printf("Week 41 Day 5: Testing CUDA Extensions\n\n");
    
    printf("Testing Strategy:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ 1. Correctness: Compare against PyTorch reference                 ║\n");
    printf("║ 2. Numerics: torch.allclose() with appropriate tolerance          ║\n");
    printf("║ 3. Edge cases: Empty tensors, large tensors, various dtypes       ║\n");
    printf("║ 4. Performance: Benchmark against baseline                        ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Test Template (test_kernel.py):\n");
    printf("```python\n");
    printf("import torch\n");
    printf("import pytest\n");
    printf("import my_cuda_ext\n");
    printf("\n");
    printf("def test_correctness():\n");
    printf("    x = torch.randn(1024, 512, device='cuda')\n");
    printf("    \n");
    printf("    # Reference\n");
    printf("    expected = torch.relu(x)\n");
    printf("    \n");
    printf("    # Custom implementation\n");
    printf("    result = my_cuda_ext.relu(x)\n");
    printf("    \n");
    printf("    torch.testing.assert_close(result, expected)\n");
    printf("\n");
    printf("@pytest.mark.parametrize('shape', [\n");
    printf("    (1, 1), (1024, 1024), (4096, 4096)\n");
    printf("])\n");
    printf("def test_shapes(shape):\n");
    printf("    x = torch.randn(*shape, device='cuda')\n");
    printf("    result = my_cuda_ext.relu(x)\n");
    printf("    assert result.shape == x.shape\n");
    printf("\n");
    printf("def test_dtypes():\n");
    printf("    for dtype in [torch.float32, torch.float16]:\n");
    printf("        x = torch.randn(1024, device='cuda', dtype=dtype)\n");
    printf("        result = my_cuda_ext.relu(x)\n");
    printf("        assert result.dtype == dtype\n");
    printf("```\n");
    
    return 0;
}
