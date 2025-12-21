#pragma once
/**
 * assertions.h - Test assertions
 */

#include "test_framework.h"
#include <cmath>
#include <cstdio>
#include <sstream>

namespace test {

// Basic assertions
#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        std::ostringstream ss; \
        ss << "ASSERT_TRUE failed: " << #cond << " at " << __FILE__ << ":" << __LINE__; \
        TestRegistry::instance().mark_failed(ss.str()); \
        return; \
    } \
} while(0)

#define ASSERT_FALSE(cond) do { \
    if (cond) { \
        std::ostringstream ss; \
        ss << "ASSERT_FALSE failed: " << #cond << " at " << __FILE__ << ":" << __LINE__; \
        TestRegistry::instance().mark_failed(ss.str()); \
        return; \
    } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    auto va = (a); auto vb = (b); \
    if (va != vb) { \
        std::ostringstream ss; \
        ss << "ASSERT_EQ failed: " << #a << " (" << va << ") != " << #b << " (" << vb << ")"; \
        ss << " at " << __FILE__ << ":" << __LINE__; \
        TestRegistry::instance().mark_failed(ss.str()); \
        return; \
    } \
} while(0)

#define ASSERT_NE(a, b) do { \
    auto va = (a); auto vb = (b); \
    if (va == vb) { \
        std::ostringstream ss; \
        ss << "ASSERT_NE failed: " << #a << " (" << va << ") == " << #b << " (" << vb << ")"; \
        ss << " at " << __FILE__ << ":" << __LINE__; \
        TestRegistry::instance().mark_failed(ss.str()); \
        return; \
    } \
} while(0)

// Float comparisons with tolerance
#define ASSERT_NEAR(a, b, tol) do { \
    auto va = (a); auto vb = (b); auto vtol = (tol); \
    if (std::abs(va - vb) > vtol) { \
        std::ostringstream ss; \
        ss << "ASSERT_NEAR failed: |" << #a << " - " << #b << "| = |" << va << " - " << vb << "| = "; \
        ss << std::abs(va - vb) << " > " << vtol; \
        ss << " at " << __FILE__ << ":" << __LINE__; \
        TestRegistry::instance().mark_failed(ss.str()); \
        return; \
    } \
} while(0)

// Array comparison
template<typename T>
inline bool arrays_near(const T* a, const T* b, size_t n, T tol, size_t* fail_idx = nullptr) {
    for (size_t i = 0; i < n; i++) {
        if (std::abs(a[i] - b[i]) > tol) {
            if (fail_idx) *fail_idx = i;
            return false;
        }
    }
    return true;
}

#define ASSERT_ARRAYS_NEAR(a, b, n, tol) do { \
    size_t fail_idx = 0; \
    if (!test::arrays_near(a, b, n, tol, &fail_idx)) { \
        std::ostringstream ss; \
        ss << "ASSERT_ARRAYS_NEAR failed at index " << fail_idx; \
        ss << ": " << (a)[fail_idx] << " vs " << (b)[fail_idx]; \
        ss << " at " << __FILE__ << ":" << __LINE__; \
        TestRegistry::instance().mark_failed(ss.str()); \
        return; \
    } \
} while(0)

}  // namespace test
