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
echo "Bottleneck analysis commands:"
echo ""
echo "1. Identify bottlenecks:"
echo "   ncu --metrics \\"
echo "     dram__throughput.avg.pct_of_peak_sustained_elapsed,\\"
echo "     sm__throughput.avg.pct_of_peak_sustained_elapsed \\"
echo "     ./build/bottleneck_demo"
echo ""
echo "2. Run optimization examples:"
echo "   ./build/optimization_examples"
echo ""
echo "3. Full analysis:"
echo "   ncu --set full -o bottleneck_report ./build/bottleneck_demo"
