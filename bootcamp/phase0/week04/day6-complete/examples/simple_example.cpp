/**
 * simple_example.cpp - Simple usage example
 */

#include "myproject/myproject.h"
#include <cstdio>
#include <vector>

int main() {
    printf("MyProject Simple Example\n");
    printf("========================\n\n");
    
    // Initialize library
    if (!myproject::initialize()) {
        fprintf(stderr, "Failed to initialize\n");
        return 1;
    }
    
    printf("Using: %s\n\n", myproject::get_device_info());
    
    // Example 1: Vector addition
    printf("Example 1: Vector Addition\n");
    const int n = 10;
    std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<float> b = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    std::vector<float> c(n);
    
    myproject::vector_add(c.data(), a.data(), b.data(), n);
    
    printf("  a: ");
    for (int i = 0; i < n; i++) printf("%.0f ", a[i]);
    printf("\n  b: ");
    for (int i = 0; i < n; i++) printf("%.0f ", b[i]);
    printf("\n  c: ");
    for (int i = 0; i < n; i++) printf("%.0f ", c[i]);
    printf("\n\n");
    
    // Example 2: Reduction
    printf("Example 2: Reduction\n");
    float sum = myproject::reduce_sum(a.data(), n);
    printf("  Sum of a: %.0f\n\n", sum);
    
    // Example 3: Matrix multiply
    printf("Example 3: Matrix Multiplication\n");
    std::vector<float> A = {1, 2, 3, 4};  // 2x2
    std::vector<float> B = {5, 6, 7, 8};  // 2x2
    std::vector<float> C(4);
    
    myproject::matmul(C.data(), A.data(), B.data(), 2, 2, 2);
    
    printf("  A = [%.0f, %.0f; %.0f, %.0f]\n", A[0], A[1], A[2], A[3]);
    printf("  B = [%.0f, %.0f; %.0f, %.0f]\n", B[0], B[1], B[2], B[3]);
    printf("  C = [%.0f, %.0f; %.0f, %.0f]\n", C[0], C[1], C[2], C[3]);
    
    myproject::cleanup();
    return 0;
}
