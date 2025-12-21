/**
 * test_main.cpp - Test runner
 */

#include "myproject/myproject.h"
#include <cstdio>
#include <cmath>
#include <vector>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    void test_##name(); \
    struct TestRunner_##name { \
        TestRunner_##name() { run_tests.push_back({#name, test_##name}); } \
    } runner_##name; \
    void test_##name()

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        printf("  FAILED: %s\n", #cond); \
        return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol) do { \
    if (std::abs((a) - (b)) > (tol)) { \
        printf("  FAILED: %s != %s (%.6f vs %.6f)\n", #a, #b, (double)(a), (double)(b)); \
        return; \
    } \
} while(0)

struct TestCase {
    const char* name;
    void (*func)();
};

static std::vector<TestCase> run_tests;

// Tests
TEST(VectorAdd_Basic) {
    const int n = 1000;
    std::vector<float> a(n), b(n), c(n);
    
    for (int i = 0; i < n; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    
    myproject::vector_add(c.data(), a.data(), b.data(), n);
    
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(c[i], 3.0f, 1e-5f);
    }
    tests_passed++;
}

TEST(VectorScale_Basic) {
    const int n = 1000;
    std::vector<float> in(n), out(n);
    
    for (int i = 0; i < n; i++) {
        in[i] = static_cast<float>(i);
    }
    
    myproject::vector_scale(out.data(), in.data(), 2.0f, n);
    
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(out[i], in[i] * 2.0f, 1e-5f);
    }
    tests_passed++;
}

TEST(ReduceSum_Basic) {
    const int n = 1000;
    std::vector<float> data(n, 1.0f);
    
    float result = myproject::reduce_sum(data.data(), n);
    ASSERT_NEAR(result, 1000.0f, 1e-2f);
    tests_passed++;
}

TEST(ReduceSum_Large) {
    const int n = 1 << 20;
    std::vector<float> data(n, 1.0f);
    
    float result = myproject::reduce_sum(data.data(), n);
    ASSERT_NEAR(result, static_cast<float>(n), n * 1e-5f);
    tests_passed++;
}

TEST(MatMul_Identity) {
    const int n = 4;
    std::vector<float> A = {1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            0, 0, 0, 1};
    std::vector<float> B = {1, 2, 3, 4,
                            5, 6, 7, 8,
                            9, 10, 11, 12,
                            13, 14, 15, 16};
    std::vector<float> C(16);
    
    myproject::matmul(C.data(), A.data(), B.data(), n, n, n);
    
    for (int i = 0; i < 16; i++) {
        ASSERT_NEAR(C[i], B[i], 1e-4f);
    }
    tests_passed++;
}

int main() {
    printf("MyProject Tests\n");
    printf("===============\n\n");
    
    if (!myproject::initialize(0)) {
        fprintf(stderr, "Failed to initialize CUDA\n");
        return 1;
    }
    
    printf("Device: %s\n\n", myproject::get_device_info());
    
    for (const auto& tc : run_tests) {
        printf("[ RUN      ] %s\n", tc.name);
        tests_run++;
        int before = tests_passed;
        tc.func();
        if (tests_passed > before) {
            printf("[       OK ] %s\n", tc.name);
        } else {
            printf("[   FAILED ] %s\n", tc.name);
        }
    }
    
    printf("\n===============\n");
    printf("Tests: %d/%d passed\n", tests_passed, tests_run);
    
    myproject::cleanup();
    
    return (tests_passed == tests_run) ? 0 : 1;
}
