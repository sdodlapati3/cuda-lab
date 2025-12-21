/**
 * test_reduction.cpp - Reduction tests
 */

#include "test/test_framework.h"
#include "test/assertions.h"
#include "test/cuda_test.cuh"
#include <vector>
#include <numeric>

TEST(Reduction_SmallArray) {
    const int n = 100;
    std::vector<float> data(n, 1.0f);
    
    float result = test::cuda::reduce_sum(data.data(), n);
    
    ASSERT_NEAR(result, 100.0f, 1e-3f);
}

TEST(Reduction_LargeArray) {
    const int n = 1000000;
    std::vector<float> data(n, 1.0f);
    
    float result = test::cuda::reduce_sum(data.data(), n);
    
    // With float, there will be some accumulation error
    ASSERT_NEAR(result, static_cast<float>(n), n * 1e-5f);
}

TEST(Reduction_SequentialValues) {
    const int n = 1000;
    std::vector<float> data(n);
    
    for (int i = 0; i < n; i++) {
        data[i] = static_cast<float>(i + 1);
    }
    
    float result = test::cuda::reduce_sum(data.data(), n);
    float expected = static_cast<float>(n * (n + 1) / 2);  // Sum of 1 to n
    
    ASSERT_NEAR(result, expected, 1e-1f);  // Tolerance for float accumulation
}

TEST(Reduction_PowerOfTwo) {
    const int n = 1024;  // Power of 2
    std::vector<float> data(n, 2.0f);
    
    float result = test::cuda::reduce_sum(data.data(), n);
    
    ASSERT_NEAR(result, 2048.0f, 1e-2f);
}

TEST(Reduction_NonPowerOfTwo) {
    const int n = 1000;  // Not power of 2
    std::vector<float> data(n, 3.0f);
    
    float result = test::cuda::reduce_sum(data.data(), n);
    
    ASSERT_NEAR(result, 3000.0f, 1e-2f);
}

TEST(Reduction_ZeroArray) {
    const int n = 1000;
    std::vector<float> data(n, 0.0f);
    
    float result = test::cuda::reduce_sum(data.data(), n);
    
    ASSERT_NEAR(result, 0.0f, 1e-5f);
}
