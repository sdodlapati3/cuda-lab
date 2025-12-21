#!/bin/bash
set -e

ARCH=${1:-80}
BUILD_TYPE=${2:-Release}

echo "Building for sm_${ARCH} in ${BUILD_TYPE} mode..."
mkdir -p build
cd build
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_CUDA_ARCHITECTURES=${ARCH} \
    ..
ninja
cd ..

echo ""
echo "Build complete!"
echo ""
echo "Debugging workflow:"
echo ""
echo "1. Find races:"
echo "   compute-sanitizer --tool racecheck ./build/buggy_app"
echo ""
echo "2. Find memory errors:"
echo "   compute-sanitizer --tool memcheck ./build/buggy_app"
echo ""
echo "3. Find memory leaks:"
echo "   compute-sanitizer --leak-check full ./build/buggy_app"
echo ""
echo "4. Debug interactively:"
echo "   cuda-gdb ./build/buggy_app"
echo ""
echo "5. Verify fixed version:"
echo "   compute-sanitizer ./build/debugged_app"
echo ""
echo "For debug build (slower but more info):"
echo "   ./build.sh ${ARCH} Debug"
