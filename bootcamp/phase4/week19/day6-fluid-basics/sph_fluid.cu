/**
 * Smoothed Particle Hydrodynamics (SPH) Fluid Simulation
 * 
 * Demonstrates:
 * - SPH kernel functions (poly6, spiky)
 * - Density and pressure computation
 * - Pressure gradient and viscosity forces
 * - Simple dam break scenario
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <cmath>

#define BLOCK_SIZE 256

// SPH Parameters
#define NUM_PARTICLES 10000
#define SMOOTHING_RADIUS 0.04f
#define REST_DENSITY 1000.0f
#define GAS_CONSTANT 2000.0f
#define VISCOSITY 0.01f
#define PARTICLE_MASS 0.02f
#define DT 0.0001f
#define NUM_STEPS 1000
#define GRAVITY -9.81f

// Spatial grid parameters
#define CELL_SIZE 0.04f
#define GRID_DIM 32

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

struct Particle {
    float x, y, z;       // Position
    float vx, vy, vz;    // Velocity
    float density;       // Computed density
    float pressure;      // Computed pressure
};

// Poly6 kernel for density
__device__ float poly6(float r2, float h) {
    float h2 = h * h;
    if (r2 > h2) return 0.0f;
    float diff = h2 - r2;
    float h9 = h2 * h2 * h2 * h2 * h;
    return 315.0f / (64.0f * M_PI * h9) * diff * diff * diff;
}

// Spiky kernel gradient for pressure
__device__ void spikyGrad(float dx, float dy, float dz, float r, float h,
                           float* gx, float* gy, float* gz) {
    if (r > h || r < 1e-6f) {
        *gx = *gy = *gz = 0.0f;
        return;
    }
    float h6 = h * h * h * h * h * h;
    float coeff = -45.0f / (M_PI * h6) * (h - r) * (h - r) / r;
    *gx = coeff * dx;
    *gy = coeff * dy;
    *gz = coeff * dz;
}

// Viscosity kernel Laplacian
__device__ float viscosityLaplacian(float r, float h) {
    if (r > h) return 0.0f;
    float h6 = h * h * h * h * h * h;
    return 45.0f / (M_PI * h6) * (h - r);
}

// Hash function
__device__ int hashPosition(float x, float y, float z, float cellSize, int gridDim) {
    int cx = (int)floorf((x + gridDim * cellSize * 0.5f) / cellSize);
    int cy = (int)floorf((y + gridDim * cellSize * 0.5f) / cellSize);
    int cz = (int)floorf((z + gridDim * cellSize * 0.5f) / cellSize);
    cx = max(0, min(gridDim - 1, cx));
    cy = max(0, min(gridDim - 1, cy));
    cz = max(0, min(gridDim - 1, cz));
    return cx + cy * gridDim + cz * gridDim * gridDim;
}

// Compute hashes
__global__ void computeHashes(Particle* particles, int* particleHash, int* particleIdx, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    particleHash[idx] = hashPosition(particles[idx].x, particles[idx].y, particles[idx].z, 
                                      CELL_SIZE, GRID_DIM);
    particleIdx[idx] = idx;
}

// Find cell boundaries
__global__ void findCellBounds(int* cellStart, int* cellEnd, int* particleHash, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int hash = particleHash[idx];
    if (idx == 0 || hash != particleHash[idx - 1]) cellStart[hash] = idx;
    if (idx == n - 1 || hash != particleHash[idx + 1]) cellEnd[hash] = idx + 1;
}

// Compute density and pressure
__global__ void computeDensityPressure(Particle* particles, int* particleIdx,
                                         int* cellStart, int* cellEnd, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float px = particles[idx].x;
    float py = particles[idx].y;
    float pz = particles[idx].z;
    
    float density = 0.0f;
    float h = SMOOTHING_RADIUS;
    
    int myCellX = (int)floorf((px + GRID_DIM * CELL_SIZE * 0.5f) / CELL_SIZE);
    int myCellY = (int)floorf((py + GRID_DIM * CELL_SIZE * 0.5f) / CELL_SIZE);
    int myCellZ = (int)floorf((pz + GRID_DIM * CELL_SIZE * 0.5f) / CELL_SIZE);
    
    // Sum contributions from neighbors
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = myCellX + dx;
                int ny = myCellY + dy;
                int nz = myCellZ + dz;
                
                if (nx < 0 || nx >= GRID_DIM ||
                    ny < 0 || ny >= GRID_DIM ||
                    nz < 0 || nz >= GRID_DIM) continue;
                
                int cellHash = nx + ny * GRID_DIM + nz * GRID_DIM * GRID_DIM;
                int start = cellStart[cellHash];
                int end = cellEnd[cellHash];
                
                if (start < 0) continue;
                
                for (int j = start; j < end; j++) {
                    int neighborIdx = particleIdx[j];
                    
                    float nx = particles[neighborIdx].x;
                    float ny = particles[neighborIdx].y;
                    float nz = particles[neighborIdx].z;
                    
                    float dx = px - nx;
                    float dy = py - ny;
                    float dz = pz - nz;
                    float r2 = dx*dx + dy*dy + dz*dz;
                    
                    density += PARTICLE_MASS * poly6(r2, h);
                }
            }
        }
    }
    
    particles[idx].density = density;
    // Equation of state: pressure from density
    particles[idx].pressure = GAS_CONSTANT * (density - REST_DENSITY);
}

// Compute forces and integrate
__global__ void computeForces(Particle* particles, int* particleIdx,
                               int* cellStart, int* cellEnd, int n, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float px = particles[idx].x;
    float py = particles[idx].y;
    float pz = particles[idx].z;
    float vx = particles[idx].vx;
    float vy = particles[idx].vy;
    float vz = particles[idx].vz;
    float myDensity = particles[idx].density;
    float myPressure = particles[idx].pressure;
    
    float h = SMOOTHING_RADIUS;
    
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    
    int myCellX = (int)floorf((px + GRID_DIM * CELL_SIZE * 0.5f) / CELL_SIZE);
    int myCellY = (int)floorf((py + GRID_DIM * CELL_SIZE * 0.5f) / CELL_SIZE);
    int myCellZ = (int)floorf((pz + GRID_DIM * CELL_SIZE * 0.5f) / CELL_SIZE);
    
    // Sum forces from neighbors
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int ncx = myCellX + dx;
                int ncy = myCellY + dy;
                int ncz = myCellZ + dz;
                
                if (ncx < 0 || ncx >= GRID_DIM ||
                    ncy < 0 || ncy >= GRID_DIM ||
                    ncz < 0 || ncz >= GRID_DIM) continue;
                
                int cellHash = ncx + ncy * GRID_DIM + ncz * GRID_DIM * GRID_DIM;
                int start = cellStart[cellHash];
                int end = cellEnd[cellHash];
                
                if (start < 0) continue;
                
                for (int j = start; j < end; j++) {
                    int neighborIdx = particleIdx[j];
                    if (neighborIdx == idx) continue;
                    
                    float nx = particles[neighborIdx].x;
                    float ny = particles[neighborIdx].y;
                    float nz = particles[neighborIdx].z;
                    float nvx = particles[neighborIdx].vx;
                    float nvy = particles[neighborIdx].vy;
                    float nvz = particles[neighborIdx].vz;
                    float neighborDensity = particles[neighborIdx].density;
                    float neighborPressure = particles[neighborIdx].pressure;
                    
                    float diffX = px - nx;
                    float diffY = py - ny;
                    float diffZ = pz - nz;
                    float r = sqrtf(diffX*diffX + diffY*diffY + diffZ*diffZ);
                    
                    if (r < h && r > 1e-6f && neighborDensity > 1e-6f) {
                        // Pressure force
                        float gx, gy, gz;
                        spikyGrad(diffX, diffY, diffZ, r, h, &gx, &gy, &gz);
                        float pressureTerm = PARTICLE_MASS * (myPressure + neighborPressure) / (2.0f * neighborDensity);
                        fx -= pressureTerm * gx;
                        fy -= pressureTerm * gy;
                        fz -= pressureTerm * gz;
                        
                        // Viscosity force
                        float viscLap = viscosityLaplacian(r, h);
                        float viscTerm = VISCOSITY * PARTICLE_MASS / neighborDensity * viscLap;
                        fx += viscTerm * (nvx - vx);
                        fy += viscTerm * (nvy - vy);
                        fz += viscTerm * (nvz - vz);
                    }
                }
            }
        }
    }
    
    // Gravity
    fy += GRAVITY * myDensity;
    
    // Acceleration
    float ax = fx / myDensity;
    float ay = fy / myDensity;
    float az = fz / myDensity;
    
    // Integration
    vx += ax * dt;
    vy += ay * dt;
    vz += az * dt;
    
    px += vx * dt;
    py += vy * dt;
    pz += vz * dt;
    
    // Simple boundary conditions (box)
    float boxSize = GRID_DIM * CELL_SIZE * 0.4f;
    float damping = 0.3f;
    
    if (px < -boxSize) { px = -boxSize; vx *= -damping; }
    if (px > boxSize) { px = boxSize; vx *= -damping; }
    if (py < -boxSize) { py = -boxSize; vy *= -damping; }
    if (py > boxSize) { py = boxSize; vy *= -damping; }
    if (pz < -boxSize) { pz = -boxSize; vz *= -damping; }
    if (pz > boxSize) { pz = boxSize; vz *= -damping; }
    
    particles[idx].x = px;
    particles[idx].y = py;
    particles[idx].z = pz;
    particles[idx].vx = vx;
    particles[idx].vy = vy;
    particles[idx].vz = vz;
}

// CPU sort helper
void sortByHash(int* particleHash, int* particleIdx, int n) {
    std::vector<std::pair<int, int>> pairs(n);
    std::vector<int> h_hash(n), h_idx(n);
    cudaMemcpy(h_hash.data(), particleHash, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idx.data(), particleIdx, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) pairs[i] = {h_hash[i], h_idx[i]};
    std::sort(pairs.begin(), pairs.end());
    for (int i = 0; i < n; i++) { h_hash[i] = pairs[i].first; h_idx[i] = pairs[i].second; }
    cudaMemcpy(particleHash, h_hash.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(particleIdx, h_idx.data(), n * sizeof(int), cudaMemcpyHostToDevice);
}

int main() {
    printf("=== SPH Fluid Simulation ===\n");
    printf("Particles: %d\n", NUM_PARTICLES);
    printf("Smoothing radius: %.3f\n", SMOOTHING_RADIUS);
    printf("Timestep: %.5f\n", DT);
    printf("Steps: %d\n\n", NUM_STEPS);
    
    // Allocate particles
    Particle* d_particles;
    cudaMalloc(&d_particles, NUM_PARTICLES * sizeof(Particle));
    
    // Initialize - dam break configuration
    std::vector<Particle> h_particles(NUM_PARTICLES);
    float spacing = SMOOTHING_RADIUS * 0.5f;
    int particlesPerDim = (int)cbrtf(NUM_PARTICLES);
    
    int idx = 0;
    for (int z = 0; z < particlesPerDim && idx < NUM_PARTICLES; z++) {
        for (int y = 0; y < particlesPerDim && idx < NUM_PARTICLES; y++) {
            for (int x = 0; x < particlesPerDim && idx < NUM_PARTICLES; x++) {
                h_particles[idx].x = -0.3f + x * spacing;
                h_particles[idx].y = -0.3f + y * spacing;
                h_particles[idx].z = -0.3f + z * spacing;
                h_particles[idx].vx = h_particles[idx].vy = h_particles[idx].vz = 0.0f;
                h_particles[idx].density = REST_DENSITY;
                h_particles[idx].pressure = 0.0f;
                idx++;
            }
        }
    }
    
    cudaMemcpy(d_particles, h_particles.data(), NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);
    
    // Allocate grid
    int numCells = GRID_DIM * GRID_DIM * GRID_DIM;
    int *d_cellStart, *d_cellEnd, *d_particleHash, *d_particleIdx;
    cudaMalloc(&d_cellStart, numCells * sizeof(int));
    cudaMalloc(&d_cellEnd, numCells * sizeof(int));
    cudaMalloc(&d_particleHash, NUM_PARTICLES * sizeof(int));
    cudaMalloc(&d_particleIdx, NUM_PARTICLES * sizeof(int));
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("Simulating...\n");
    cudaEventRecord(start);
    
    for (int step = 0; step < NUM_STEPS; step++) {
        // Build spatial hash
        computeHashes<<<grid, block>>>(d_particles, d_particleHash, d_particleIdx, NUM_PARTICLES);
        cudaDeviceSynchronize();
        sortByHash(d_particleHash, d_particleIdx, NUM_PARTICLES);
        
        cudaMemset(d_cellStart, -1, numCells * sizeof(int));
        cudaMemset(d_cellEnd, -1, numCells * sizeof(int));
        findCellBounds<<<grid, block>>>(d_cellStart, d_cellEnd, d_particleHash, NUM_PARTICLES);
        
        // Compute density and pressure
        computeDensityPressure<<<grid, block>>>(d_particles, d_particleIdx, 
                                                  d_cellStart, d_cellEnd, NUM_PARTICLES);
        
        // Compute forces and integrate
        computeForces<<<grid, block>>>(d_particles, d_particleIdx,
                                        d_cellStart, d_cellEnd, NUM_PARTICLES, DT);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float totalMs;
    cudaEventElapsedTime(&totalMs, start, stop);
    
    // Get final state
    cudaMemcpy(h_particles.data(), d_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);
    
    // Calculate average density
    float avgDensity = 0.0f;
    float minY = 1e10f, maxY = -1e10f;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        avgDensity += h_particles[i].density;
        minY = fminf(minY, h_particles[i].y);
        maxY = fmaxf(maxY, h_particles[i].y);
    }
    avgDensity /= NUM_PARTICLES;
    
    printf("\nResults:\n");
    printf("--------\n");
    printf("Simulation time: %.2f ms\n", totalMs);
    printf("Time per step: %.3f ms\n", totalMs / NUM_STEPS);
    printf("Steps per second: %.0f\n", NUM_STEPS / (totalMs / 1000.0f));
    printf("Average density: %.1f (rest: %.1f)\n", avgDensity, REST_DENSITY);
    printf("Y range: [%.3f, %.3f]\n", minY, maxY);
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_particles);
    cudaFree(d_cellStart);
    cudaFree(d_cellEnd);
    cudaFree(d_particleHash);
    cudaFree(d_particleIdx);
    
    return 0;
}
