#pragma once
/**
 * test_framework.h - Simple test framework
 */

#include <string>
#include <vector>
#include <functional>

namespace test {

// Test result
struct TestResult {
    std::string name;
    bool passed;
    std::string message;
    double time_ms;
};

// Test case
struct TestCase {
    std::string name;
    std::function<void()> func;
};

// Test registry (singleton)
class TestRegistry {
public:
    static TestRegistry& instance();
    
    void register_test(const std::string& name, std::function<void()> func);
    
    int run_all(const std::string& filter = "");
    
    void set_current_test(const std::string& name);
    void mark_failed(const std::string& msg);
    
    bool current_passed() const { return current_passed_; }
    
private:
    TestRegistry() = default;
    std::vector<TestCase> tests_;
    std::vector<TestResult> results_;
    
    std::string current_test_;
    bool current_passed_ = true;
    std::string fail_message_;
};

// Test registration macro
#define TEST(name) \
    static void test_##name(); \
    static struct TestRegistrar_##name { \
        TestRegistrar_##name() { \
            test::TestRegistry::instance().register_test(#name, test_##name); \
        } \
    } registrar_##name; \
    static void test_##name()

}  // namespace test
