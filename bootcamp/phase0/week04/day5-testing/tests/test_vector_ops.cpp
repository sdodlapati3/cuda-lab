/**
 * test_vector_ops.cpp - Vector operation tests
 */

#include "test/test_framework.h"
#include "test/assertions.h"
#include "test/cuda_test.cuh"
#include <vector>

TEST(VectorAdd_BasicTest) {
    const int n = 1000;
    std::vector<float> a(n), b(n), c(n), expected(n);
    
    for (int i = 0; i < n; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(n - i);
        expected[i] = static_cast<float>(n);  // a[i] + b[i] = n
    }
    
    test::cuda::vector_add(c.data(), a.data(), b.data(), n);
    
    ASSERT_ARRAYS_NEAR(c.data(), expected.data(), (size_t)n, 1e-5f);
}

TEST(VectorAdd_ZeroElements) {
    const int n = 1000;
    std::vector<float> a(n, 0.0f), b(n, 0.0f), c(n), expected(n, 0.0f);
    
    test::cuda::vector_add(c.data(), a.data(), b.data(), n);
    
    ASSERT_ARRAYS_NEAR(c.data(), expected.data(), (size_t)n, 1e-5f);
}

TEST(VectorAdd_NegativeNumbers) {
    const int n = 1000;
    std::vector<float> a(n), b(n), c(n), expected(n);
    
    for (int i = 0; i < n; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(-i);
        expected[i] = 0.0f;
    }
    
    test::cuda::vector_add(c.data(), a.data(), b.data(), n);
    
    ASSERT_ARRAYS_NEAR(c.data(), expected.data(), (size_t)n, 1e-5f);
}

TEST(VectorScale_BasicTest) {
    const int n = 1000;
    const float scalar = 2.5f;
    std::vector<float> in(n), out(n), expected(n);
    
    for (int i = 0; i < n; i++) {
        in[i] = static_cast<float>(i);
        expected[i] = in[i] * scalar;
    }
    
    test::cuda::vector_scale(out.data(), in.data(), scalar, n);
    
    ASSERT_ARRAYS_NEAR(out.data(), expected.data(), (size_t)n, 1e-4f);
}

TEST(VectorScale_ZeroScalar) {
    const int n = 1000;
    std::vector<float> in(n), out(n), expected(n, 0.0f);
    
    for (int i = 0; i < n; i++) {
        in[i] = static_cast<float>(i);
    }
    
    test::cuda::vector_scale(out.data(), in.data(), 0.0f, n);
    
    ASSERT_ARRAYS_NEAR(out.data(), expected.data(), (size_t)n, 1e-5f);
}
