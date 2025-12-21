/**
 * Tiled N-Body Simulation with Shared Memory Optimization
 * 
 * Uses shared memory tiles to reduce global memory bandwidth requirements.
 * Achieves significant speedup over naive O(N²) approach.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 256
#define SOFTENING 1e-9f
#define NUM_BODIES 16384
#define NUM_STEPS 10
#define DT 0.001f

// Naive kernel for comparison (from Day 1)
__global__ void computeForces_naive(float4* pos, float3* acc, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float4 myPos = pos[i];
    float3 a = make_float3(0.0f, 0.0f, 0.0f);
    
    for (int j = 0; j < n; j++) {
        float4 otherPos = pos[j];  // Global memory load for each j
        
        float dx = otherPos.x - myPos.x;
        float dy = otherPos.y - myPos.y;
        float dz = otherPos.z - myPos.z;
        
        float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;
        float s = otherPos.w * invDist3;  // mass * 1/r³
        
        a.x += s * dx;
        a.y += s * dy;
        a.z += s * dz;
    }
    
    acc[i] = a;
}

// Tiled kernel with shared memory optimization
__global__ void computeForces_tiled(float4* pos, float3* acc, int n) {
    extern __shared__ float4 sharedPos[];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float4 myPos;
    if (i < n) {
        myPos = pos[i];
    } else {
        myPos = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    
    float3 a = make_float3(0.0f, 0.0f, 0.0f);
    
    // Process bodies in tiles
    int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int tile = 0; tile < numTiles; tile++) {
        // Load one body per thread into shared memory
        int loadIdx = tile * BLOCK_SIZE + threadIdx.x;
        if (loadIdx < n) {
            sharedPos[threadIdx.x] = pos[loadIdx];
        } else {
            sharedPos[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        
        __syncthreads();
        
        // Compute interactions with all bodies in this tile
        #pragma unroll 32
        for (int j = 0; j < BLOCK_SIZE; j++) {
            float4 otherPos = sharedPos[j];  // Fast shared memory access
            
            float dx = otherPos.x - myPos.x;
            float dy = otherPos.y - myPos.y;
            float dz = otherPos.z - myPos.z;
            
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;
            float s = otherPos.w * invDist3;
            
            a.x += s * dx;
            a.y += s * dy;
            a.z += s * dz;
        }
        
        __syncthreads();
    }
    
    if (i < n) {
        acc[i] = a;
    }
}

__global__ void integrate(float4* pos, float4* vel, float3* acc, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float4 p = pos[i];
    float4 v = vel[i];
    float3 a = acc[i];
    
    // Leapfrog integration
    v.x += a.x * dt;
    v.y += a.y * dt;
    v.z += a.z * dt;
    
    p.x += v.x * dt;
    p.y += v.y * dt;
    p.z += v.z * dt;
    
    pos[i] = p;
    vel[i] = v;
}

void initBodies(float4* pos, float4* vel, int n) {
    for (int i = 0; i < n; i++) {
        // Random positions in unit cube
        pos[i].x = (float)rand() / RAND_MAX - 0.5f;
        pos[i].y = (float)rand() / RAND_MAX - 0.5f;
        pos[i].z = (float)rand() / RAND_MAX - 0.5f;
        pos[i].w = 1.0f / n;  // Equal masses
        
        // Small random velocities
        vel[i].x = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        vel[i].y = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        vel[i].z = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        vel[i].w = 0.0f;
    }
}

int main() {
    printf("=== Tiled N-Body Simulation ===\n");
    printf("Bodies: %d, Interactions per step: %.2f billion\n", 
           NUM_BODIES, (double)NUM_BODIES * NUM_BODIES / 1e9);
    printf("Block size (tile size): %d\n\n", BLOCK_SIZE);
    
    // Allocate memory
    size_t posSize = NUM_BODIES * sizeof(float4);
    size_t velSize = NUM_BODIES * sizeof(float4);
    size_t accSize = NUM_BODIES * sizeof(float3);
    
    float4 *h_pos = (float4*)malloc(posSize);
    float4 *h_vel = (float4*)malloc(velSize);
    
    float4 *d_pos, *d_vel;
    float3 *d_acc;
    cudaMalloc(&d_pos, posSize);
    cudaMalloc(&d_vel, velSize);
    cudaMalloc(&d_acc, accSize);
    
    // Initialize bodies
    srand(42);
    initBodies(h_pos, h_vel, NUM_BODIES);
    
    cudaMemcpy(d_pos, h_pos, posSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, h_vel, velSize, cudaMemcpyHostToDevice);
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_BODIES + BLOCK_SIZE - 1) / BLOCK_SIZE);
    size_t sharedMemSize = BLOCK_SIZE * sizeof(float4);
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm-up
    computeForces_tiled<<<grid, block, sharedMemSize>>>(d_pos, d_acc, NUM_BODIES);
    cudaDeviceSynchronize();
    
    // Benchmark naive kernel
    cudaEventRecord(start);
    for (int step = 0; step < NUM_STEPS; step++) {
        computeForces_naive<<<grid, block>>>(d_pos, d_acc, NUM_BODIES);
        integrate<<<grid, block>>>(d_pos, d_vel, d_acc, NUM_BODIES, DT);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float naiveMs;
    cudaEventElapsedTime(&naiveMs, start, stop);
    
    // Reset positions
    cudaMemcpy(d_pos, h_pos, posSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, h_vel, velSize, cudaMemcpyHostToDevice);
    
    // Benchmark tiled kernel
    cudaEventRecord(start);
    for (int step = 0; step < NUM_STEPS; step++) {
        computeForces_tiled<<<grid, block, sharedMemSize>>>(d_pos, d_acc, NUM_BODIES);
        integrate<<<grid, block>>>(d_pos, d_vel, d_acc, NUM_BODIES, DT);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float tiledMs;
    cudaEventElapsedTime(&tiledMs, start, stop);
    
    // Calculate GFLOPS (20 FLOPS per interaction)
    double interactions = (double)NUM_BODIES * NUM_BODIES * NUM_STEPS;
    double naiveGflops = (interactions * 20) / (naiveMs / 1000.0) / 1e9;
    double tiledGflops = (interactions * 20) / (tiledMs / 1000.0) / 1e9;
    
    printf("Performance Results:\n");
    printf("-------------------\n");
    printf("Naive kernel:  %.2f ms (%.2f GFLOPS)\n", naiveMs, naiveGflops);
    printf("Tiled kernel:  %.2f ms (%.2f GFLOPS)\n", tiledMs, tiledGflops);
    printf("Speedup: %.2fx\n", naiveMs / tiledMs);
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_acc);
    free(h_pos);
    free(h_vel);
    
    return 0;
}
