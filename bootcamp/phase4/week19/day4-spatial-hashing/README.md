# Day 4: Spatial Hashing

## Learning Objectives
- Understand uniform grid spatial partitioning
- Implement hash-based particle lookup
- Build neighbor lists on GPU
- Optimize for GPU parallel construction

## Why Spatial Hashing?

### N-Body Problem
- Naive: O(N²) interactions
- With spatial hashing: O(N × k) where k = avg neighbors

### Key Idea
- Divide space into uniform grid cells
- Hash particle position to cell index
- Only check particles in nearby cells

## Hash Function
```cpp
// 3D position to 1D cell index
__device__ int hashPosition(float3 pos, float cellSize, int3 gridDim) {
    int cx = (int)floorf(pos.x / cellSize);
    int cy = (int)floorf(pos.y / cellSize);
    int cz = (int)floorf(pos.z / cellSize);
    
    // Wrap for periodic boundaries (optional)
    cx = (cx % gridDim.x + gridDim.x) % gridDim.x;
    cy = (cy % gridDim.y + gridDim.y) % gridDim.y;
    cz = (cz % gridDim.z + gridDim.z) % gridDim.z;
    
    return cx + cy * gridDim.x + cz * gridDim.x * gridDim.y;
}
```

## GPU-Friendly Grid Construction

### Algorithm
1. **Compute cell indices** for all particles
2. **Sort particles** by cell index (using radix sort)
3. **Find cell boundaries** (start/end indices)

### Data Structures
```cpp
struct SpatialGrid {
    int* cellStart;   // First particle in each cell
    int* cellEnd;     // Last particle in each cell  
    int* particleIdx; // Sorted particle indices
    int* cellHash;    // Cell hash for each particle
};
```

## Neighbor Search Pattern
```cpp
// For each particle, check 27 neighboring cells (3x3x3)
for (int dz = -1; dz <= 1; dz++) {
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int neighborCell = hash(myCell + (dx, dy, dz));
            // Iterate particles in neighborCell
        }
    }
}
```

## Exercises
1. Implement spatial hash grid construction
2. Build neighbor lists for all particles
3. Use spatial hashing for N-body simulation
4. Compare performance vs naive approach
