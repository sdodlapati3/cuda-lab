# Day 1: N-Body Basics

## Learning Objectives
- Implement naive N-body gravitational simulation
- Understand O(N²) parallelization
- Measure baseline performance

## Key Formula
```
acceleration_i = Σ G * m_j * (pos_j - pos_i) / (|pos_j - pos_i|³ + softening)
```

The softening factor prevents singularities when particles are close.
