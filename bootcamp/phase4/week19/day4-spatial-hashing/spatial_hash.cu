/**
 * Spatial Hashing for Efficient Neighbor Search
 * 
 * Demonstrates:
 * - Uniform grid spatial partitioning
 * - Hash-based particle lookup
 * - Cell-based neighbor finding
 * - Significant speedup for local interactions
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#define BLOCK_SIZE 256
#define NUM_PARTICLES 100000
#define CELL_SIZE 0.1f
#define INTERACTION_RADIUS 0.1f
#define GRID_DIM 64  // 64x64x64 grid

// Spatial hash grid structure
struct SpatialGrid {
    int* cellStart;    // First particle index in each cell
    int* cellEnd;      // Last particle index in each cell
    int* particleHash; // Hash (cell index) for each particle
    int* particleIdx;  // Sorted particle indices
    int numCells;
};

// Hash function: 3D position to 1D cell index
__device__ __host__ int hashPosition(float x, float y, float z, float cellSize, int gridDim) {
    int cx = (int)floorf((x + gridDim * cellSize * 0.5f) / cellSize);
    int cy = (int)floorf((y + gridDim * cellSize * 0.5f) / cellSize);
    int cz = (int)floorf((z + gridDim * cellSize * 0.5f) / cellSize);
    
    // Clamp to grid bounds
    cx = max(0, min(gridDim - 1, cx));
    cy = max(0, min(gridDim - 1, cy));
    cz = max(0, min(gridDim - 1, cz));
    
    return cx + cy * gridDim + cz * gridDim * gridDim;
}

// Compute hash for all particles
__global__ void computeHashes(float* posX, float* posY, float* posZ,
                               int* particleHash, int* particleIdx,
                               int n, float cellSize, int gridDim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int hash = hashPosition(posX[idx], posY[idx], posZ[idx], cellSize, gridDim);
    particleHash[idx] = hash;
    particleIdx[idx] = idx;
}

// Find cell boundaries after sorting
__global__ void findCellBounds(int* cellStart, int* cellEnd,
                                int* particleHash, int n, int numCells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int hash = particleHash[idx];
    
    // Check if this is the first particle in its cell
    if (idx == 0 || hash != particleHash[idx - 1]) {
        cellStart[hash] = idx;
    }
    
    // Check if this is the last particle in its cell
    if (idx == n - 1 || hash != particleHash[idx + 1]) {
        cellEnd[hash] = idx + 1;
    }
}

// Naive neighbor count (O(N²) - for comparison)
__global__ void countNeighborsNaive(float* posX, float* posY, float* posZ,
                                      int* neighborCount, int n, float radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float myX = posX[idx];
    float myY = posY[idx];
    float myZ = posZ[idx];
    float radiusSqr = radius * radius;
    
    int count = 0;
    for (int j = 0; j < n; j++) {
        if (j == idx) continue;
        
        float dx = posX[j] - myX;
        float dy = posY[j] - myY;
        float dz = posZ[j] - myZ;
        float distSqr = dx*dx + dy*dy + dz*dz;
        
        if (distSqr < radiusSqr) {
            count++;
        }
    }
    
    neighborCount[idx] = count;
}

// Spatial hash neighbor count (O(N × k))
__global__ void countNeighborsSpatialHash(float* posX, float* posY, float* posZ,
                                            int* particleIdx, int* cellStart, int* cellEnd,
                                            int* neighborCount, int n,
                                            float radius, float cellSize, int gridDim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float myX = posX[idx];
    float myY = posY[idx];
    float myZ = posZ[idx];
    float radiusSqr = radius * radius;
    
    int count = 0;
    
    // Get my cell coordinates
    int myCellX = (int)floorf((myX + gridDim * cellSize * 0.5f) / cellSize);
    int myCellY = (int)floorf((myY + gridDim * cellSize * 0.5f) / cellSize);
    int myCellZ = (int)floorf((myZ + gridDim * cellSize * 0.5f) / cellSize);
    
    // Check 3x3x3 neighboring cells
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = myCellX + dx;
                int ny = myCellY + dy;
                int nz = myCellZ + dz;
                
                // Skip out-of-bounds cells
                if (nx < 0 || nx >= gridDim ||
                    ny < 0 || ny >= gridDim ||
                    nz < 0 || nz >= gridDim) continue;
                
                int cellHash = nx + ny * gridDim + nz * gridDim * gridDim;
                int start = cellStart[cellHash];
                int end = cellEnd[cellHash];
                
                if (start < 0) continue;  // Empty cell
                
                // Check all particles in this cell
                for (int j = start; j < end; j++) {
                    int neighborIdx = particleIdx[j];
                    if (neighborIdx == idx) continue;
                    
                    float dx = posX[neighborIdx] - myX;
                    float dy = posY[neighborIdx] - myY;
                    float dz = posZ[neighborIdx] - myZ;
                    float distSqr = dx*dx + dy*dy + dz*dz;
                    
                    if (distSqr < radiusSqr) {
                        count++;
                    }
                }
            }
        }
    }
    
    neighborCount[idx] = count;
}

// Simple counting sort by hash (for demonstration)
void sortByHash(int* particleHash, int* particleIdx, int n) {
    // Create index array for CPU sort
    std::vector<std::pair<int, int>> pairs(n);
    
    std::vector<int> h_hash(n);
    std::vector<int> h_idx(n);
    cudaMemcpy(h_hash.data(), particleHash, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idx.data(), particleIdx, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n; i++) {
        pairs[i] = {h_hash[i], h_idx[i]};
    }
    
    std::sort(pairs.begin(), pairs.end());
    
    for (int i = 0; i < n; i++) {
        h_hash[i] = pairs[i].first;
        h_idx[i] = pairs[i].second;
    }
    
    cudaMemcpy(particleHash, h_hash.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(particleIdx, h_idx.data(), n * sizeof(int), cudaMemcpyHostToDevice);
}

int main() {
    printf("=== Spatial Hashing for Neighbor Search ===\n");
    printf("Particles: %d\n", NUM_PARTICLES);
    printf("Grid: %dx%dx%d = %d cells\n", GRID_DIM, GRID_DIM, GRID_DIM, GRID_DIM*GRID_DIM*GRID_DIM);
    printf("Interaction radius: %.3f\n\n", INTERACTION_RADIUS);
    
    // Allocate positions (SoA)
    float *d_posX, *d_posY, *d_posZ;
    cudaMalloc(&d_posX, NUM_PARTICLES * sizeof(float));
    cudaMalloc(&d_posY, NUM_PARTICLES * sizeof(float));
    cudaMalloc(&d_posZ, NUM_PARTICLES * sizeof(float));
    
    // Initialize random positions on host
    std::vector<float> h_posX(NUM_PARTICLES);
    std::vector<float> h_posY(NUM_PARTICLES);
    std::vector<float> h_posZ(NUM_PARTICLES);
    
    srand(42);
    float domainSize = GRID_DIM * CELL_SIZE * 0.4f;  // Keep particles within grid
    for (int i = 0; i < NUM_PARTICLES; i++) {
        h_posX[i] = ((float)rand() / RAND_MAX - 0.5f) * domainSize;
        h_posY[i] = ((float)rand() / RAND_MAX - 0.5f) * domainSize;
        h_posZ[i] = ((float)rand() / RAND_MAX - 0.5f) * domainSize;
    }
    
    cudaMemcpy(d_posX, h_posX.data(), NUM_PARTICLES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posY, h_posY.data(), NUM_PARTICLES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posZ, h_posZ.data(), NUM_PARTICLES * sizeof(float), cudaMemcpyHostToDevice);
    
    // Allocate spatial grid
    int numCells = GRID_DIM * GRID_DIM * GRID_DIM;
    SpatialGrid grid;
    grid.numCells = numCells;
    cudaMalloc(&grid.cellStart, numCells * sizeof(int));
    cudaMalloc(&grid.cellEnd, numCells * sizeof(int));
    cudaMalloc(&grid.particleHash, NUM_PARTICLES * sizeof(int));
    cudaMalloc(&grid.particleIdx, NUM_PARTICLES * sizeof(int));
    
    int* d_neighborCount;
    cudaMalloc(&d_neighborCount, NUM_PARTICLES * sizeof(int));
    
    dim3 block(BLOCK_SIZE);
    dim3 gridParticles((NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 gridCells((numCells + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Build spatial hash grid
    printf("Building spatial hash grid...\n");
    cudaEventRecord(start);
    
    // Step 1: Compute hashes
    computeHashes<<<gridParticles, block>>>(d_posX, d_posY, d_posZ,
                                             grid.particleHash, grid.particleIdx,
                                             NUM_PARTICLES, CELL_SIZE, GRID_DIM);
    
    // Step 2: Sort by hash (using CPU sort for simplicity)
    cudaDeviceSynchronize();
    sortByHash(grid.particleHash, grid.particleIdx, NUM_PARTICLES);
    
    // Step 3: Initialize cell bounds to -1
    cudaMemset(grid.cellStart, -1, numCells * sizeof(int));
    cudaMemset(grid.cellEnd, -1, numCells * sizeof(int));
    
    // Step 4: Find cell boundaries
    findCellBounds<<<gridParticles, block>>>(grid.cellStart, grid.cellEnd,
                                              grid.particleHash, NUM_PARTICLES, numCells);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float buildMs;
    cudaEventElapsedTime(&buildMs, start, stop);
    printf("Grid build time: %.2f ms\n\n", buildMs);
    
    // Benchmark spatial hash neighbor search
    cudaEventRecord(start);
    countNeighborsSpatialHash<<<gridParticles, block>>>(d_posX, d_posY, d_posZ,
                                                          grid.particleIdx, grid.cellStart, grid.cellEnd,
                                                          d_neighborCount, NUM_PARTICLES,
                                                          INTERACTION_RADIUS, CELL_SIZE, GRID_DIM);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float hashMs;
    cudaEventElapsedTime(&hashMs, start, stop);
    
    // Get average neighbor count
    std::vector<int> h_neighborCount(NUM_PARTICLES);
    cudaMemcpy(h_neighborCount.data(), d_neighborCount, NUM_PARTICLES * sizeof(int), cudaMemcpyDeviceToHost);
    
    long long totalNeighbors = 0;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        totalNeighbors += h_neighborCount[i];
    }
    float avgNeighbors = (float)totalNeighbors / NUM_PARTICLES;
    
    printf("Spatial Hash Neighbor Search:\n");
    printf("  Time: %.2f ms\n", hashMs);
    printf("  Average neighbors per particle: %.1f\n", avgNeighbors);
    printf("  Estimated speedup vs naive: %.0fx (N/k = %d/%.1f)\n",
           (float)NUM_PARTICLES / avgNeighbors, NUM_PARTICLES, avgNeighbors);
    
    // Note: Naive benchmark would take too long for 100K particles
    printf("\n(Naive O(N²) benchmark skipped - would take ~%.0f seconds)\n",
           (double)NUM_PARTICLES * NUM_PARTICLES / 1e9);
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_posX);
    cudaFree(d_posY);
    cudaFree(d_posZ);
    cudaFree(grid.cellStart);
    cudaFree(grid.cellEnd);
    cudaFree(grid.particleHash);
    cudaFree(grid.particleIdx);
    cudaFree(d_neighborCount);
    
    return 0;
}
