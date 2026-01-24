#!/bin/bash
# build-container.sh - Build Singularity container
#
# Usage:
#   ./build-container.sh cuda-pytorch.def pytorch.sif
#   ./build-container.sh cuda-pytorch.def  # Auto-names output

set -e

DEF_FILE="${1:-cuda-pytorch.def}"
SIF_FILE="${2:-${DEF_FILE%.def}.sif}"

echo "Building container from: $DEF_FILE"
echo "Output: $SIF_FILE"

# Check if Singularity/Apptainer is available
if command -v apptainer &> /dev/null; then
    BUILDER="apptainer"
elif command -v singularity &> /dev/null; then
    BUILDER="singularity"
else
    echo "Error: Neither Singularity nor Apptainer found"
    exit 1
fi

echo "Using builder: $BUILDER"

# Build options
BUILD_OPTS=""

# Add --fakeroot if not running as root
if [ "$(id -u)" -ne 0 ]; then
    echo "Building with --fakeroot (non-root build)"
    BUILD_OPTS="--fakeroot"
fi

# Build the container
echo "Starting build..."
$BUILDER build $BUILD_OPTS "$SIF_FILE" "$DEF_FILE"

echo ""
echo "Build complete: $SIF_FILE"
echo ""
echo "Test the container:"
echo "  $BUILDER exec --nv $SIF_FILE python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
echo "Interactive shell:"
echo "  $BUILDER shell --nv $SIF_FILE"
