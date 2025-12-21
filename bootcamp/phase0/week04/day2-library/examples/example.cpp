/**
 * example.cpp - Example usage of MyCudaLib
 * 
 * Note: This is a .cpp file, not .cu - the library hides CUDA details!
 */

#include <mylib/mylib.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main() {
    printf("MyCudaLib Example\n");
    printf("=================\n\n");
    
    // Initialize library
    if (!mylib::initialize()) {
        fprintf(stderr, "Failed to initialize: %s\n", mylib::get_last_error());
        return 1;
    }
    
    const int N = 1000000;
    float* a = new float[N];
    float* b = new float[N];
    float* c = new float[N];
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    
    // Test vector_add
    printf("Testing vector_add...\n");
    mylib::vector_add(a, b, c, N);
    
    bool correct = true;
    for (int i = 0; i < N && correct; i++) {
        if (fabs(c[i] - 3.0f) > 1e-5f) correct = false;
    }
    printf("  Result: %s\n", correct ? "PASS" : "FAIL");
    
    // Test vector_scale
    printf("Testing vector_scale...\n");
    mylib::vector_scale(a, c, 5.0f, N);
    
    correct = true;
    for (int i = 0; i < N && correct; i++) {
        if (fabs(c[i] - 5.0f) > 1e-5f) correct = false;
    }
    printf("  Result: %s\n", correct ? "PASS" : "FAIL");
    
    // Test dot_product
    printf("Testing dot_product...\n");
    float dot = mylib::dot_product(a, b, N);
    float expected = N * 1.0f * 2.0f;
    correct = (fabs(dot - expected) / expected < 0.001f);
    printf("  Got %.0f, expected %.0f: %s\n", dot, expected, correct ? "PASS" : "FAIL");
    
    // Cleanup
    delete[] a;
    delete[] b;
    delete[] c;
    mylib::cleanup();
    
    printf("\nDone!\n");
    return 0;
}
