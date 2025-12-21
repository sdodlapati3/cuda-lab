# Day 5: Collision Detection

## Learning Objectives
- Understand broad phase vs narrow phase collision detection
- Implement AABB (Axis-Aligned Bounding Box) tests
- Use spatial hashing for broad phase
- Handle sphere-sphere collisions with response

## Collision Detection Phases

### 1. Broad Phase
- Goal: Quickly eliminate pairs that can't possibly collide
- Methods: Spatial hashing, BVH trees, sweep-and-prune
- Output: Candidate collision pairs

### 2. Narrow Phase
- Goal: Precise collision test on candidates
- Methods: AABB, sphere-sphere, GJK algorithm
- Output: Contact points, normals, penetration depth

## AABB Test
```cpp
__device__ bool aabbIntersect(AABB a, AABB b) {
    return (a.min.x <= b.max.x && a.max.x >= b.min.x) &&
           (a.min.y <= b.max.y && a.max.y >= b.min.y) &&
           (a.min.z <= b.max.z && a.max.z >= b.min.z);
}
```

## Sphere-Sphere Collision
```cpp
__device__ bool sphereIntersect(float3 p1, float r1, float3 p2, float r2) {
    float3 d = make_float3(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
    float distSqr = d.x*d.x + d.y*d.y + d.z*d.z;
    float radiusSum = r1 + r2;
    return distSqr < radiusSum * radiusSum;
}
```

## Collision Response
1. Calculate penetration depth
2. Compute collision normal
3. Apply impulse based on masses and velocities
4. Separate overlapping objects

## GPU Considerations
- Parallel broad phase is straightforward
- Narrow phase needs careful memory management
- Use atomic operations for collision counting
- Sort collisions for cache efficiency

## Exercises
1. Implement AABB collision detection
2. Add sphere-sphere narrow phase
3. Integrate with spatial hashing from Day 4
4. Add elastic collision response
