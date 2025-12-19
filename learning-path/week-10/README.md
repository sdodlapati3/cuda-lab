# Week 10: CUDA Graphs

## Learning Philosophy

> **CUDA C++ First, Python/Numba as Optional Backup**

All notebooks show CUDA C++ code as the PRIMARY implementation. Python/Numba is provided optionally for quick interactive testing in Colab.

## Overview

Learn to capture and replay GPU workflows for minimal launch overhead.

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | Graph Basics | What are graphs, stream capture, graph instantiation |
| 2 | Explicit Graphs | Building graphs node by node, dependencies |
| 3 | Graph Updates | Modifying parameters without rebuilding |
| 4 | Graph Optimization | Best practices, when to use graphs |
| 5 | Practice & Quiz | Exercises + checkpoint assessment |

## Prerequisites
- Week 9: CUDA Streams & Concurrency
- Understanding of async operations
- Event-based synchronization

## Key Skills
- [ ] Capture stream operations into graphs
- [ ] Build explicit graphs with node dependencies
- [ ] Update graph parameters efficiently
- [ ] Profile graph vs stream performance
- [ ] Identify good use cases for graphs
