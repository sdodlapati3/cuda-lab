/**
 * GPU Collision Detection
 * 
 * Demonstrates:
 * - Broad phase with spatial hashing
 * - Narrow phase sphere-sphere tests
 * - Elastic collision response
 * - Parallel collision pair detection
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

#define BLOCK_SIZE 256
#define NUM_SPHERES 50000
#define SPHERE_RADIUS 0.02f
#define CELL_SIZE 0.05f  // Slightly larger than diameter
#define GRID_DIM 32
#define MAX_COLLISIONS_PER_SPHERE 32

struct Sphere {
    float x, y, z;      // Position
    float vx, vy, vz;   // Velocity
    float radius;
    float mass;
};

// Collision pair
struct CollisionPair {
    int i, j;           // Sphere indices
    float nx, ny, nz;   // Collision normal
    float penetration;  // Penetration depth
};

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

// Compute hashes for all spheres
__global__ void computeHashes(Sphere* spheres, int* particleHash, int* particleIdx, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int hash = hashPosition(spheres[idx].x, spheres[idx].y, spheres[idx].z, 
                            CELL_SIZE, GRID_DIM);
    particleHash[idx] = hash;
    particleIdx[idx] = idx;
}

// Find cell boundaries
__global__ void findCellBounds(int* cellStart, int* cellEnd, int* particleHash, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int hash = particleHash[idx];
    
    if (idx == 0 || hash != particleHash[idx - 1]) {
        cellStart[hash] = idx;
    }
    if (idx == n - 1 || hash != particleHash[idx + 1]) {
        cellEnd[hash] = idx + 1;
    }
}

// Sphere-sphere collision test
__device__ bool sphereCollision(Sphere a, Sphere b, float* nx, float* ny, float* nz, float* penetration) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float dz = b.z - a.z;
    
    float distSqr = dx*dx + dy*dy + dz*dz;
    float radiusSum = a.radius + b.radius;
    
    if (distSqr < radiusSum * radiusSum && distSqr > 1e-10f) {
        float dist = sqrtf(distSqr);
        *penetration = radiusSum - dist;
        
        // Collision normal (from a to b)
        float invDist = 1.0f / dist;
        *nx = dx * invDist;
        *ny = dy * invDist;
        *nz = dz * invDist;
        
        return true;
    }
    return false;
}

// Detect collisions using spatial hashing
__global__ void detectCollisions(Sphere* spheres, int* particleIdx,
                                   int* cellStart, int* cellEnd,
                                   int* collisionCount, CollisionPair* collisions,
                                   int n, int maxCollisions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Sphere me = spheres[idx];
    
    int myCellX = (int)floorf((me.x + GRID_DIM * CELL_SIZE * 0.5f) / CELL_SIZE);
    int myCellY = (int)floorf((me.y + GRID_DIM * CELL_SIZE * 0.5f) / CELL_SIZE);
    int myCellZ = (int)floorf((me.z + GRID_DIM * CELL_SIZE * 0.5f) / CELL_SIZE);
    
    // Check neighboring cells
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
                    int otherIdx = particleIdx[j];
                    if (otherIdx <= idx) continue;  // Avoid duplicate pairs
                    
                    Sphere other = spheres[otherIdx];
                    float cnx, cny, cnz, penetration;
                    
                    if (sphereCollision(me, other, &cnx, &cny, &cnz, &penetration)) {
                        int collisionIdx = atomicAdd(collisionCount, 1);
                        if (collisionIdx < maxCollisions) {
                            collisions[collisionIdx].i = idx;
                            collisions[collisionIdx].j = otherIdx;
                            collisions[collisionIdx].nx = cnx;
                            collisions[collisionIdx].ny = cny;
                            collisions[collisionIdx].nz = cnz;
                            collisions[collisionIdx].penetration = penetration;
                        }
                    }
                }
            }
        }
    }
}

// Apply collision response (elastic collision)
__global__ void resolveCollisions(Sphere* spheres, CollisionPair* collisions, int numCollisions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numCollisions) return;
    
    CollisionPair col = collisions[idx];
    Sphere& a = spheres[col.i];
    Sphere& b = spheres[col.j];
    
    // Relative velocity
    float rvx = b.vx - a.vx;
    float rvy = b.vy - a.vy;
    float rvz = b.vz - a.vz;
    
    // Relative velocity along collision normal
    float velAlongNormal = rvx * col.nx + rvy * col.ny + rvz * col.nz;
    
    // Don't resolve if velocities are separating
    if (velAlongNormal > 0) return;
    
    // Coefficient of restitution (elasticity)
    float e = 0.9f;
    
    // Impulse scalar
    float j = -(1 + e) * velAlongNormal;
    j /= 1.0f / a.mass + 1.0f / b.mass;
    
    // Apply impulse
    float impulseX = j * col.nx;
    float impulseY = j * col.ny;
    float impulseZ = j * col.nz;
    
    // Use atomic operations for thread safety
    atomicAdd(&a.vx, -impulseX / a.mass);
    atomicAdd(&a.vy, -impulseY / a.mass);
    atomicAdd(&a.vz, -impulseZ / a.mass);
    atomicAdd(&b.vx, impulseX / b.mass);
    atomicAdd(&b.vy, impulseY / b.mass);
    atomicAdd(&b.vz, impulseZ / b.mass);
    
    // Separate spheres
    float separateX = col.penetration * 0.5f * col.nx;
    float separateY = col.penetration * 0.5f * col.ny;
    float separateZ = col.penetration * 0.5f * col.nz;
    
    atomicAdd(&a.x, -separateX);
    atomicAdd(&a.y, -separateY);
    atomicAdd(&a.z, -separateZ);
    atomicAdd(&b.x, separateX);
    atomicAdd(&b.y, separateY);
    atomicAdd(&b.z, separateZ);
}

// CPU sort for simplicity (would use thrust::sort in production)
void sortByHash(int* particleHash, int* particleIdx, int n) {
    std::vector<std::pair<int, int>> pairs(n);
    std::vector<int> h_hash(n), h_idx(n);
    
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
    printf("=== GPU Collision Detection ===\n");
    printf("Spheres: %d\n", NUM_SPHERES);
    printf("Sphere radius: %.3f\n", SPHERE_RADIUS);
    printf("Grid: %d³ cells\n\n", GRID_DIM);
    
    // Allocate spheres
    Sphere* d_spheres;
    cudaMalloc(&d_spheres, NUM_SPHERES * sizeof(Sphere));
    
    // Initialize spheres on host
    std::vector<Sphere> h_spheres(NUM_SPHERES);
    srand(42);
    float domainSize = GRID_DIM * CELL_SIZE * 0.4f;
    
    for (int i = 0; i < NUM_SPHERES; i++) {
        h_spheres[i].x = ((float)rand() / RAND_MAX - 0.5f) * domainSize;
        h_spheres[i].y = ((float)rand() / RAND_MAX - 0.5f) * domainSize;
        h_spheres[i].z = ((float)rand() / RAND_MAX - 0.5f) * domainSize;
        h_spheres[i].vx = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        h_spheres[i].vy = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        h_spheres[i].vz = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        h_spheres[i].radius = SPHERE_RADIUS;
        h_spheres[i].mass = 1.0f;
    }
    
    cudaMemcpy(d_spheres, h_spheres.data(), NUM_SPHERES * sizeof(Sphere), cudaMemcpyHostToDevice);
    
    // Allocate spatial grid
    int numCells = GRID_DIM * GRID_DIM * GRID_DIM;
    int* d_cellStart, *d_cellEnd, *d_particleHash, *d_particleIdx;
    cudaMalloc(&d_cellStart, numCells * sizeof(int));
    cudaMalloc(&d_cellEnd, numCells * sizeof(int));
    cudaMalloc(&d_particleHash, NUM_SPHERES * sizeof(int));
    cudaMalloc(&d_particleIdx, NUM_SPHERES * sizeof(int));
    
    // Allocate collision output
    int maxCollisions = NUM_SPHERES * MAX_COLLISIONS_PER_SPHERE;
    CollisionPair* d_collisions;
    int* d_collisionCount;
    cudaMalloc(&d_collisions, maxCollisions * sizeof(CollisionPair));
    cudaMalloc(&d_collisionCount, sizeof(int));
    
    dim3 block(BLOCK_SIZE);
    dim3 gridSpheres((NUM_SPHERES + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("Running collision detection...\n");
    cudaEventRecord(start);
    
    // Step 1: Build spatial hash
    computeHashes<<<gridSpheres, block>>>(d_spheres, d_particleHash, d_particleIdx, NUM_SPHERES);
    cudaDeviceSynchronize();
    sortByHash(d_particleHash, d_particleIdx, NUM_SPHERES);
    
    cudaMemset(d_cellStart, -1, numCells * sizeof(int));
    cudaMemset(d_cellEnd, -1, numCells * sizeof(int));
    findCellBounds<<<gridSpheres, block>>>(d_cellStart, d_cellEnd, d_particleHash, NUM_SPHERES);
    
    // Step 2: Detect collisions
    cudaMemset(d_collisionCount, 0, sizeof(int));
    detectCollisions<<<gridSpheres, block>>>(d_spheres, d_particleIdx,
                                               d_cellStart, d_cellEnd,
                                               d_collisionCount, d_collisions,
                                               NUM_SPHERES, maxCollisions);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float detectMs;
    cudaEventElapsedTime(&detectMs, start, stop);
    
    // Get collision count
    int numCollisions;
    cudaMemcpy(&numCollisions, d_collisionCount, sizeof(int), cudaMemcpyDeviceToHost);
    numCollisions = min(numCollisions, maxCollisions);
    
    // Step 3: Resolve collisions
    cudaEventRecord(start);
    if (numCollisions > 0) {
        dim3 gridCollisions((numCollisions + BLOCK_SIZE - 1) / BLOCK_SIZE);
        resolveCollisions<<<gridCollisions, block>>>(d_spheres, d_collisions, numCollisions);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float resolveMs;
    cudaEventElapsedTime(&resolveMs, start, stop);
    
    printf("\nResults:\n");
    printf("--------\n");
    printf("Collisions detected: %d\n", numCollisions);
    printf("Detection time: %.2f ms\n", detectMs);
    printf("Resolution time: %.2f ms\n", resolveMs);
    printf("Total time: %.2f ms\n", detectMs + resolveMs);
    printf("Pairs tested (approx): %.0f million vs %.0f billion naive\n",
           (double)numCollisions * 27 / 1e6,
           (double)NUM_SPHERES * NUM_SPHERES / 2 / 1e9);
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_spheres);
    cudaFree(d_cellStart);
    cudaFree(d_cellEnd);
    cudaFree(d_particleHash);
    cudaFree(d_particleIdx);
    cudaFree(d_collisions);
    cudaFree(d_collisionCount);
    
    return 0;
}
