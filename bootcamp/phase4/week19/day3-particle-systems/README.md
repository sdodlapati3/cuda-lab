# Day 3: Particle Systems

## Learning Objectives
- Design flexible particle system data structures
- Implement multiple force types (gravity, drag, springs)
- Handle particle emission and death
- Create visual effects with particles

## Particle System Components

### Particle State
```cpp
struct Particle {
    float3 position;
    float3 velocity;
    float  age;        // Time since spawn
    float  lifetime;   // Max lifetime
    int    type;       // Particle category
};
```

### Force Generators
1. **Constant Forces**: Gravity, wind
2. **Position-Dependent**: Springs, drag
3. **Velocity-Dependent**: Damping
4. **Inter-Particle**: Cohesion, avoidance

### Particle Lifecycle
1. **Emission**: Spawn new particles at sources
2. **Simulation**: Apply forces, integrate
3. **Death**: Remove expired particles
4. **Compaction**: Pack active particles

## GPU-Friendly Design

### Structure of Arrays (SoA)
```cpp
// Better for GPU coalesced access
float* positions_x;  // All x coordinates
float* positions_y;  // All y coordinates
float* positions_z;  // All z coordinates
float* velocities_x;
// ... etc
```

### Particle Pooling
- Pre-allocate maximum particles
- Use atomic counters for emission
- Stream compaction for removal

## Exercises
1. Implement basic particle system
2. Add multiple force types
3. Create particle emitter with spawn rate
4. Implement particle death and respawn
