/*
 * Day 1: N-Body Basics
 * 
 * Naive O(N²) gravitational N-body simulation.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

#define CHECK_CUDA(call) do { cudaError_t e = call; if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } } while(0)

struct float4 { float x, y, z, w; };

__global__ void computeForces_naive(
    const float4* __restrict__ pos,
    float3* __restrict__ acc,
    int n, float softening)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float3 ai = {0.0f, 0.0f, 0.0f};
    float4 pi = pos[i];
    
    for (int j = 0; j < n; j++) {
        float4 pj = pos[j];
        
        float dx = pj.x - pi.x;
        float dy = pj.y - pi.y;
        float dz = pj.z - pi.z;
        
        float distSq = dx*dx + dy*dy + dz*dz + softening;
        float invDist = rsqrtf(distSq);
        float invDist3 = invDist * invDist * invDist;
        float s = pj.w * invDist3;  // w = mass
        
        ai.x += dx * s;
        ai.y += dy * s;
        ai.z += dz * s;
    }
    
    acc[i] = ai;
}

__global__ void integrate(
    float4* pos,
    float4* vel,
    const float3* acc,
    float dt, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float4 p = pos[i];
    float4 v = vel[i];
    float3 a = acc[i];
    
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
        float theta = 2.0f * M_PI * rand() / RAND_MAX;
        float r = 10.0f * rand() / RAND_MAX;
        pos[i].x = r * cosf(theta);
        pos[i].y = r * sinf(theta);
        pos[i].z = 2.0f * (rand() / (float)RAND_MAX - 0.5f);
        pos[i].w = 1.0f;  // mass
        
        vel[i].x = -pos[i].y * 0.1f;
        vel[i].y = pos[i].x * 0.1f;
        vel[i].z = 0.0f;
        vel[i].w = 0.0f;
    }
}

int main() {
    printf("=== Day 1: N-Body Basics ===\n\n");
    
    const int N = 16384;
    const float dt = 0.01f;
    const float softening = 0.001f;
    const int iterations = 10;
    
    printf("Particles: %d\n", N);
    printf("Interactions per step: %.1f million\n", (N * (float)N) / 1e6f);
    
    // Allocate
    float4 *h_pos = new float4[N];
    float4 *h_vel = new float4[N];
    initBodies(h_pos, h_vel, N);
    
    float4 *d_pos, *d_vel;
    float3 *d_acc;
    CHECK_CUDA(cudaMalloc(&d_pos, N * sizeof(float4)));
    CHECK_CUDA(cudaMalloc(&d_vel, N * sizeof(float4)));
    CHECK_CUDA(cudaMalloc(&d_acc, N * sizeof(float3)));
    
    CHECK_CUDA(cudaMemcpy(d_pos, h_pos, N * sizeof(float4), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vel, h_vel, N * sizeof(float4), cudaMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((N + 255) / 256);
    
    // Warmup
    computeForces_naive<<<grid, block>>>(d_pos, d_acc, N, softening);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int iter = 0; iter < iterations; iter++) {
        computeForces_naive<<<grid, block>>>(d_pos, d_acc, N, softening);
        integrate<<<grid, block>>>(d_pos, d_vel, d_acc, dt, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float msPerStep = ms / iterations;
    float gflops = (N * (float)N * 20) / (msPerStep * 1e6f);  // ~20 FLOPs per interaction
    
    printf("\nPerformance:\n");
    printf("  Time per step: %.2f ms\n", msPerStep);
    printf("  GFLOPS: %.1f\n", gflops);
    printf("  Interactions/sec: %.2f billion\n", (N * (float)N) / (msPerStep * 1e6f));
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_pos));
    CHECK_CUDA(cudaFree(d_vel));
    CHECK_CUDA(cudaFree(d_acc));
    delete[] h_pos;
    delete[] h_vel;
    
    printf("\n=== Day 1 Complete ===\n");
    return 0;
}
