#!/bin/bash
#SBATCH --job-name=ncu-app-replay
#SBATCH --partition=h100flex-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/ncu-app-replay-%j.out
#SBATCH --error=logs/ncu-app-replay-%j.err

module load python3

NSIGHT_ENV="$HOME/envs/nsight_tools"
NCU="$NSIGHT_ENV/nsight-compute-2025.4.1/ncu"
REPORT_DIR="$HOME/cuda-lab/profiling-lab/reports"

cd /tmp

# Simple kernel
cat > simple_add.cu << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 1<<20;
    size_t size = n * sizeof(float);
    
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    for(int i=0; i<n; i++) { h_a[i]=1.0f; h_b[i]=2.0f; }
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    printf("Result: c[0]=%f c[100]=%f (should be 3.0)\n", h_c[0], h_c[100]);
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
EOF

echo "=== Compiling ==="
crun -p $NSIGHT_ENV nvcc -o simple_add simple_add.cu

echo "=== Test run ==="
./simple_add

echo ""
echo "=== NCU with application replay mode ==="
$NCU --replay-mode application \
     --set basic \
     -o $REPORT_DIR/simple_add_report \
     --force-overwrite \
     ./simple_add

echo ""
echo "=== Check reports ==="
ls -la $REPORT_DIR/*.ncu-rep 2>/dev/null || echo "No .ncu-rep files"
