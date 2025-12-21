#pragma once
/**
 * utils.h - Utility functions
 */

#include <cstdio>

namespace app {

// Progress bar
inline void print_progress(int current, int total) {
    int percent = (current * 100) / total;
    printf("\rProgress: [");
    for (int i = 0; i < 50; i++) {
        if (i < percent / 2) printf("=");
        else if (i == percent / 2) printf(">");
        else printf(" ");
    }
    printf("] %d%%", percent);
    fflush(stdout);
    if (current == total) printf("\n");
}

// Timer for CPU operations
class Timer {
public:
    void start() { start_ = clock(); }
    double elapsed_ms() const {
        return (clock() - start_) * 1000.0 / CLOCKS_PER_SEC;
    }
private:
    clock_t start_;
};

}  // namespace app
