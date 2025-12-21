# Week 19: Physics Simulation

## Overview

GPU-accelerated physics for games, visualization, and scientific computing.

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | N-Body Basics | Gravitational simulation, O(N²) |
| 2 | Tiled N-Body | Shared memory optimization |
| 3 | Particle Systems | Position, velocity, forces |
| 4 | Spatial Hashing | Grid-based acceleration |
| 5 | Collision Detection | Broad/narrow phase |
| 6 | Fluid Basics | SPH fundamentals |

## Key Concepts

### N-Body Problem
Each particle influenced by all others:
```
F_i = Σ G * m_i * m_j / |r_ij|² * r̂_ij
```
Naive: O(N²) - perfect for GPU parallelization

### Optimization Strategies
1. **Tiling**: Load particles into shared memory
2. **Spatial Structures**: Reduce N² to N*k for nearby particles
3. **Barnes-Hut**: O(N log N) with tree approximation

## Building

```bash
cd day1-nbody-basics
./build.sh
./build/nbody_basic
```
