# Exercise 07: CDP Octree Construction

## Objective
Build an octree spatial data structure using Dynamic Parallelism.

## Background
Octrees recursively subdivide 3D space into 8 octants. Each node either:
- Is a leaf (contains points directly)
- Has 8 children (subdivided further)

CDP is perfect for this because:
- Subdivision decisions happen at runtime
- Different regions may need different depths
- Parallel construction of independent subtrees

## Requirements
1. Implement point counting per octant
2. Recursively subdivide if count > threshold
3. Track tree structure in output

## Testing
```bash
make
./test.sh
```
