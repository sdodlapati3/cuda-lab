#!/bin/bash
#SBATCH --job-name=ncu-test
#SBATCH --partition=h100flex-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/ncu-test-%j.out
#SBATCH --error=logs/ncu-test-%j.err

module load python3

NSIGHT_ENV="$HOME/envs/nsight_tools"
NCU="$NSIGHT_ENV/nsight-compute-2025.4.1/ncu"

echo "=== Testing NCU with vectorAdd sample ==="
cd /tmp

# Create simple test
cat > test_kernel.cu << 'EOF'
#include <stdio.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 1024;
    float *a, *b, *c;
    cudaMallocManaged(&a, n*sizeof(float));
    cudaMallocManaged(&b, n*sizeof(float));
    cudaMallocManaged(&c, n*sizeof(float));
    
    for(int i=0; i<n; i++) { a[i]=1.0f; b[i]=2.0f; }
    
    vectorAdd<<<4, 256>>>(a, b, c, n);
    cudaDeviceSynchronize();
    
    printf("Result: c[0]=%f (should be 3.0)\n", c[0]);
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
EOF

echo "=== Compiling ==="
crun -p $NSIGHT_ENV nvcc -o test_kernel test_kernel.cu
echo ""

echo "=== Running without profiler ==="
./test_kernel
echo ""

echo "=== Running with ncu (print to stdout) ==="
$NCU --target-processes all ./test_kernel
echo ""

echo "=== Running with ncu (save report) ==="
$NCU -o /home/sdodl001/cuda-lab/profiling-lab/reports/test_kernel_report --force-overwrite ./test_kernel
echo ""

echo "=== Check report ==="
ls -la /home/sdodl001/cuda-lab/profiling-lab/reports/
