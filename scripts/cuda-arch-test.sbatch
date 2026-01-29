#!/bin/bash
#SBATCH --job-name=cuda-arch
#SBATCH --partition=h100flex-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/cuda-arch-%j.out
#SBATCH --error=logs/cuda-arch-%j.err

module load python3
NSIGHT_ENV="$HOME/envs/nsight_tools"

nvidia-smi
echo ""
cd /tmp

cat > arch_test.cu << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void simple_kernel(int *out) {
    out[0] = 42;
}

int main() {
    int deviceId;
    cudaGetDevice(&deviceId);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("SM count: %d\n", prop.multiProcessorCount);
    
    int *d_out, h_out = 0;
    cudaError_t err;
    
    err = cudaMalloc(&d_out, sizeof(int));
    printf("cudaMalloc: %s\n", cudaGetErrorString(err));
    
    simple_kernel<<<1, 1>>>(d_out);
    err = cudaGetLastError();
    printf("Kernel launch: %s\n", cudaGetErrorString(err));
    
    err = cudaDeviceSynchronize();
    printf("Sync: %s\n", cudaGetErrorString(err));
    
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Result: %d (expected 42)\n", h_out);
    
    cudaFree(d_out);
    return 0;
}
EOF

echo "=== Compiling for sm_90 (H100) ==="
crun -p $NSIGHT_ENV nvcc -arch=sm_90 -o arch_test arch_test.cu

echo "=== Running ==="
./arch_test
