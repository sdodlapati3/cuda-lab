# Day 6: Fluid Simulation Basics (SPH)

## Learning Objectives
- Understand Smoothed Particle Hydrodynamics (SPH)
- Implement kernel functions for SPH
- Compute density and pressure forces
- Create a simple fluid simulation

## SPH Overview

### Core Idea
- Represent fluid as particles
- Properties (density, pressure) are smoothed over nearby particles
- Forces computed from pressure gradients and viscosity

### The SPH Equation
For any field A at position r:
$$A(r) = \sum_j m_j \frac{A_j}{\rho_j} W(r - r_j, h)$$

Where:
- $m_j$ = mass of particle j
- $\rho_j$ = density at particle j
- $W$ = smoothing kernel
- $h$ = smoothing radius

## SPH Kernels

### Poly6 Kernel (for density)
```cpp
__device__ float poly6(float r, float h) {
    if (r > h) return 0.0f;
    float h2 = h * h;
    float h9 = h2 * h2 * h2 * h2 * h;
    float diff = h2 - r * r;
    return 315.0f / (64.0f * M_PI * h9) * diff * diff * diff;
}
```

### Spiky Kernel (for pressure gradient)
```cpp
__device__ float3 spikyGrad(float3 r, float dist, float h) {
    if (dist > h || dist < 1e-6f) return make_float3(0, 0, 0);
    float h6 = h * h * h * h * h * h;
    float coeff = -45.0f / (M_PI * h6) * (h - dist) * (h - dist) / dist;
    return make_float3(coeff * r.x, coeff * r.y, coeff * r.z);
}
```

## SPH Forces

### 1. Pressure Force
$$F_{pressure} = -\nabla p = -\sum_j m_j \frac{p_i + p_j}{2\rho_j} \nabla W$$

### 2. Viscosity Force
$$F_{viscosity} = \mu \nabla^2 v = \mu \sum_j m_j \frac{v_j - v_i}{\rho_j} \nabla^2 W$$

### 3. External Forces
- Gravity: $F = m \cdot g$
- Boundary forces

## Algorithm
1. Build neighbor lists (spatial hashing)
2. Compute density for each particle
3. Compute pressure from density (EOS)
4. Compute pressure gradient force
5. Compute viscosity force
6. Integrate positions and velocities

## Exercises
1. Implement SPH density computation
2. Add pressure force calculation
3. Implement viscosity damping
4. Create dam break simulation
