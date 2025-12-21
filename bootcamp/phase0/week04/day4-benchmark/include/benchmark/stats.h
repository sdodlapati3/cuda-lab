#pragma once
/**
 * stats.h - Statistical utilities
 */

#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace bench {

struct Stats {
    double mean;
    double std_dev;
    double min;
    double max;
    double median;
    
    static Stats compute(const std::vector<double>& data) {
        Stats s;
        if (data.empty()) return s;
        
        // Mean
        s.mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        
        // Std dev
        double sq_sum = 0.0;
        for (double v : data) {
            sq_sum += (v - s.mean) * (v - s.mean);
        }
        s.std_dev = std::sqrt(sq_sum / data.size());
        
        // Min/Max
        auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
        s.min = *min_it;
        s.max = *max_it;
        
        // Median
        std::vector<double> sorted = data;
        std::sort(sorted.begin(), sorted.end());
        size_t n = sorted.size();
        if (n % 2 == 0) {
            s.median = (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
        } else {
            s.median = sorted[n/2];
        }
        
        return s;
    }
};

}  // namespace bench
