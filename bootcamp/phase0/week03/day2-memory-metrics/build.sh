#!/bin/bash
set -e

ARCH=${1:-80}

echo "Building for sm_${ARCH}..."
mkdir -p build
cd build
cmake -G Ninja -DCMAKE_CUDA_ARCHITECTURES=${ARCH} ..
ninja
cd ..

echo ""
echo "Build complete!"
echo ""
echo "Memory profiling commands:"
echo ""
echo "1. Memory workload analysis:"
echo "   ncu --set memory -o memory_report ./build/memory_patterns"
echo ""
echo "2. Bandwidth test:"
echo "   ./build/bandwidth_test"
echo ""
echo "3. Key memory metrics:"
echo "   ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\\"
echo "   l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld ./build/memory_patterns"
