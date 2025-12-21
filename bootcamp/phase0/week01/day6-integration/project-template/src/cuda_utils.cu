/**
 * CUDA Utilities - Implementation file
 * 
 * This file exists to create a linkable library.
 * Most utilities are header-only (cuda_utils.cuh).
 */

#include "cuda_utils.cuh"

// Library initialization - prints device info
namespace {
    struct LibInit {
        LibInit() {
            // Uncomment to auto-print GPU info on program start
            // get_gpu_info().print();
        }
    };
    // LibInit init;  // Uncomment to enable
}
