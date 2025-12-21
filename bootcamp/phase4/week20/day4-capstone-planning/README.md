# Day 4: Capstone Project Planning

## Objective
Design a substantial CUDA project that demonstrates mastery of GPU programming concepts learned throughout Phase 4.

## Project Selection Criteria

### Required Elements
- [ ] Custom CUDA kernels (not just library calls)
- [ ] Performance optimization techniques
- [ ] Memory management considerations
- [ ] Benchmarking vs baseline

### Recommended Elements
- [ ] Multiple kernel types
- [ ] Shared memory usage
- [ ] Stream-based pipelining
- [ ] Multi-pass algorithms

## Project Ideas

### 1. Real-Time Particle System (Difficulty: ⭐⭐⭐)
Combine N-body, collision detection, and rendering
- Features: gravity, collisions, visual effects
- Techniques: spatial hashing, tiled forces
- Deliverable: Interactive demo or video

### 2. Image Processing Pipeline (Difficulty: ⭐⭐)
Chain multiple operations efficiently
- Features: load, filter, transform, save
- Techniques: kernel fusion, memory planning
- Deliverable: Command-line tool

### 3. Mini ML Inference Engine (Difficulty: ⭐⭐⭐)
Complete inference for simple model
- Features: load weights, forward pass, quantization
- Techniques: batching, layer fusion
- Deliverable: Python-callable library

### 4. Ray Marcher (Difficulty: ⭐⭐⭐⭐)
GPU-based 3D rendering
- Features: signed distance fields, lighting
- Techniques: parallel pixel computation
- Deliverable: Image or video output

### 5. Fluid Simulation (Difficulty: ⭐⭐⭐⭐)
Extend Day 6's SPH implementation
- Features: visualization, interaction
- Techniques: neighbor search, integration
- Deliverable: Simulation video

## Planning Template

### 1. Project Overview
- Name:
- Description (2-3 sentences):
- Difficulty level:

### 2. Technical Requirements
- Input format:
- Output format:
- Target performance:

### 3. Kernel Design
| Kernel | Purpose | Block Size | Shared Memory |
|--------|---------|------------|---------------|
| | | | |

### 4. Memory Layout
- Host allocations:
- Device allocations:
- Transfer strategy:

### 5. Milestones
| Day | Milestone | Deliverable |
|-----|-----------|-------------|
| 4 | Planning | This document |
| 5 | Core implementation | Working prototype |
| 6 | Optimization & docs | Final submission |

## Evaluation Criteria
1. **Correctness** (30%): Does it produce correct results?
2. **Performance** (30%): Is it faster than CPU baseline?
3. **Code Quality** (20%): Clean, readable, well-commented?
4. **Documentation** (20%): README, benchmarks, analysis?

## Exercises
1. Select a project from the list (or propose your own)
2. Complete the planning template
3. Identify potential challenges
4. Create initial file structure
