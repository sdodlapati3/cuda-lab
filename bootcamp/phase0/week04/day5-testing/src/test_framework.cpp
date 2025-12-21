/**
 * test_framework.cpp - Test framework implementation
 */

#include "test/test_framework.h"
#include <cstdio>
#include <chrono>

namespace test {

TestRegistry& TestRegistry::instance() {
    static TestRegistry inst;
    return inst;
}

void TestRegistry::register_test(const std::string& name, std::function<void()> func) {
    tests_.push_back({name, func});
}

void TestRegistry::set_current_test(const std::string& name) {
    current_test_ = name;
    current_passed_ = true;
    fail_message_.clear();
}

void TestRegistry::mark_failed(const std::string& msg) {
    current_passed_ = false;
    fail_message_ = msg;
}

int TestRegistry::run_all(const std::string& filter) {
    results_.clear();
    
    int total = 0;
    int passed = 0;
    int skipped = 0;
    
    for (const auto& tc : tests_) {
        // Apply filter
        if (!filter.empty() && tc.name.find(filter) == std::string::npos) {
            skipped++;
            continue;
        }
        
        total++;
        set_current_test(tc.name);
        
        printf("[ RUN      ] %s\n", tc.name.c_str());
        
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            tc.func();
        } catch (const std::exception& e) {
            mark_failed(std::string("Exception: ") + e.what());
        } catch (...) {
            mark_failed("Unknown exception");
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        TestResult result;
        result.name = tc.name;
        result.passed = current_passed_;
        result.message = fail_message_;
        result.time_ms = ms;
        results_.push_back(result);
        
        if (current_passed_) {
            printf("[       OK ] %s (%.3f ms)\n", tc.name.c_str(), ms);
            passed++;
        } else {
            printf("[   FAILED ] %s\n", tc.name.c_str());
            printf("             %s\n", fail_message_.c_str());
        }
    }
    
    // Summary
    printf("\n");
    printf("===================\n");
    printf("Test Summary\n");
    printf("===================\n");
    printf("Total:   %d\n", total);
    printf("Passed:  %d\n", passed);
    printf("Failed:  %d\n", total - passed);
    if (skipped > 0) {
        printf("Skipped: %d\n", skipped);
    }
    printf("\n");
    
    if (passed == total) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("SOME TESTS FAILED:\n");
        for (const auto& r : results_) {
            if (!r.passed) {
                printf("  - %s\n", r.name.c_str());
            }
        }
    }
    
    return total - passed;
}

}  // namespace test
