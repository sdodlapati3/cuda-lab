/**
 * GPU Particle System
 * 
 * Demonstrates:
 * - Structure of Arrays (SoA) layout for particles
 * - Multiple force types (gravity, drag, attractors)
 * - Particle emission with atomic counters
 * - Particle death and respawn
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_PARTICLES 1000000
#define BLOCK_SIZE 256
#define GRAVITY -9.81f
#define DRAG_COEFF 0.1f
#define DT 0.016f  // ~60 FPS

// Particle system using Structure of Arrays
struct ParticleSystem {
    // Positions
    float* x;
    float* y;
    float* z;
    
    // Velocities
    float* vx;
    float* vy;
    float* vz;
    
    // Lifecycle
    float* age;
    float* lifetime;
    int* active;  // 1 = alive, 0 = dead
    
    // Count
    int* numActive;
    
    // Random states
    curandState* randStates;
};

// Initialize random states
__global__ void initRandom(curandState* states, int n, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Emit new particles at origin with random velocities
__global__ void emitParticles(ParticleSystem ps, int emitCount, int maxParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= maxParticles) return;
    
    // Only emit if particle is dead and we haven't hit emit count
    if (ps.active[idx] == 0) {
        int emitIdx = atomicAdd(ps.numActive, 1);
        if (emitIdx < emitCount) {
            curandState localState = ps.randStates[idx];
            
            // Spawn at origin
            ps.x[idx] = 0.0f;
            ps.y[idx] = 0.0f;
            ps.z[idx] = 0.0f;
            
            // Random velocity (cone upward)
            float angle = curand_uniform(&localState) * 2.0f * 3.14159f;
            float speed = 5.0f + curand_uniform(&localState) * 10.0f;
            float spread = 0.3f;
            
            ps.vx[idx] = speed * spread * cosf(angle);
            ps.vy[idx] = speed;
            ps.vz[idx] = speed * spread * sinf(angle);
            
            // Random lifetime
            ps.age[idx] = 0.0f;
            ps.lifetime[idx] = 1.0f + curand_uniform(&localState) * 3.0f;
            ps.active[idx] = 1;
            
            ps.randStates[idx] = localState;
        }
    }
}

// Apply forces and integrate
__global__ void updateParticles(ParticleSystem ps, int n, float dt,
                                 float3 attractorPos, float attractorStrength) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || ps.active[idx] == 0) return;
    
    // Load particle state
    float px = ps.x[idx];
    float py = ps.y[idx];
    float pz = ps.z[idx];
    float vx = ps.vx[idx];
    float vy = ps.vy[idx];
    float vz = ps.vz[idx];
    
    // Force accumulator
    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;
    
    // 1. Gravity
    fy += GRAVITY;
    
    // 2. Drag (velocity-dependent)
    float speed = sqrtf(vx*vx + vy*vy + vz*vz);
    if (speed > 0.001f) {
        float dragMag = DRAG_COEFF * speed * speed;
        fx -= dragMag * vx / speed;
        fy -= dragMag * vy / speed;
        fz -= dragMag * vz / speed;
    }
    
    // 3. Attractor force
    float dx = attractorPos.x - px;
    float dy = attractorPos.y - py;
    float dz = attractorPos.z - pz;
    float dist = sqrtf(dx*dx + dy*dy + dz*dz + 0.1f);
    float attractForce = attractorStrength / (dist * dist);
    fx += attractForce * dx / dist;
    fy += attractForce * dy / dist;
    fz += attractForce * dz / dist;
    
    // Integration (Euler)
    vx += fx * dt;
    vy += fy * dt;
    vz += fz * dt;
    
    px += vx * dt;
    py += vy * dt;
    pz += vz * dt;
    
    // Floor collision
    if (py < 0.0f) {
        py = 0.0f;
        vy = -vy * 0.5f;  // Bounce with energy loss
    }
    
    // Update age
    float age = ps.age[idx] + dt;
    
    // Check death
    if (age >= ps.lifetime[idx]) {
        ps.active[idx] = 0;
    } else {
        // Store updated state
        ps.x[idx] = px;
        ps.y[idx] = py;
        ps.z[idx] = pz;
        ps.vx[idx] = vx;
        ps.vy[idx] = vy;
        ps.vz[idx] = vz;
        ps.age[idx] = age;
    }
}

// Count active particles
__global__ void countActive(int* active, int* count, int n) {
    __shared__ int localCount;
    if (threadIdx.x == 0) localCount = 0;
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && active[idx] == 1) {
        atomicAdd(&localCount, 1);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        atomicAdd(count, localCount);
    }
}

ParticleSystem allocateParticleSystem(int maxParticles) {
    ParticleSystem ps;
    
    cudaMalloc(&ps.x, maxParticles * sizeof(float));
    cudaMalloc(&ps.y, maxParticles * sizeof(float));
    cudaMalloc(&ps.z, maxParticles * sizeof(float));
    cudaMalloc(&ps.vx, maxParticles * sizeof(float));
    cudaMalloc(&ps.vy, maxParticles * sizeof(float));
    cudaMalloc(&ps.vz, maxParticles * sizeof(float));
    cudaMalloc(&ps.age, maxParticles * sizeof(float));
    cudaMalloc(&ps.lifetime, maxParticles * sizeof(float));
    cudaMalloc(&ps.active, maxParticles * sizeof(int));
    cudaMalloc(&ps.numActive, sizeof(int));
    cudaMalloc(&ps.randStates, maxParticles * sizeof(curandState));
    
    // Initialize to zero
    cudaMemset(ps.active, 0, maxParticles * sizeof(int));
    cudaMemset(ps.numActive, 0, sizeof(int));
    
    // Initialize random states
    dim3 block(BLOCK_SIZE);
    dim3 grid((maxParticles + BLOCK_SIZE - 1) / BLOCK_SIZE);
    initRandom<<<grid, block>>>(ps.randStates, maxParticles, 12345);
    
    return ps;
}

void freeParticleSystem(ParticleSystem& ps) {
    cudaFree(ps.x);
    cudaFree(ps.y);
    cudaFree(ps.z);
    cudaFree(ps.vx);
    cudaFree(ps.vy);
    cudaFree(ps.vz);
    cudaFree(ps.age);
    cudaFree(ps.lifetime);
    cudaFree(ps.active);
    cudaFree(ps.numActive);
    cudaFree(ps.randStates);
}

int main() {
    printf("=== GPU Particle System ===\n");
    printf("Max particles: %d\n\n", MAX_PARTICLES);
    
    ParticleSystem ps = allocateParticleSystem(MAX_PARTICLES);
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((MAX_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Attractor position
    float3 attractor = make_float3(0.0f, 5.0f, 0.0f);
    float attractorStrength = 50.0f;
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int emitPerFrame = 10000;
    int numFrames = 100;
    
    cudaEventRecord(start);
    
    for (int frame = 0; frame < numFrames; frame++) {
        // Reset emit counter
        cudaMemset(ps.numActive, 0, sizeof(int));
        
        // Emit new particles
        emitParticles<<<grid, block>>>(ps, emitPerFrame, MAX_PARTICLES);
        
        // Update all particles
        updateParticles<<<grid, block>>>(ps, MAX_PARTICLES, DT, attractor, attractorStrength);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float totalMs;
    cudaEventElapsedTime(&totalMs, start, stop);
    
    // Count final active particles
    int* d_count;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));
    countActive<<<grid, block>>>(ps.active, d_count, MAX_PARTICLES);
    
    int activeCount;
    cudaMemcpy(&activeCount, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Simulation Results:\n");
    printf("-------------------\n");
    printf("Frames simulated: %d\n", numFrames);
    printf("Particles emitted per frame: %d\n", emitPerFrame);
    printf("Active particles at end: %d\n", activeCount);
    printf("Total time: %.2f ms\n", totalMs);
    printf("Time per frame: %.3f ms\n", totalMs / numFrames);
    printf("Particle updates per second: %.2f million\n", 
           (double)MAX_PARTICLES * numFrames / (totalMs / 1000.0) / 1e6);
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_count);
    freeParticleSystem(ps);
    
    return 0;
}
