# Exercise 05: CDP Patterns

## Objective
Implement common Dynamic Parallelism patterns on the GPU.

## Background
CUDA Dynamic Parallelism (CDP) allows kernels to launch other kernels from device code. This enables:
- Recursive algorithms on GPU
- Adaptive workloads
- Irregular parallelism

## Requirements
1. Compile with `-rdc=true -lcudadevrt`
2. Implement recursive parallel sum using CDP
3. Use proper parent-child synchronization

## Pattern: Recursive Parallel Sum
```
sum([1,2,3,4,5,6,7,8])
├── sum([1,2,3,4]) + sum([5,6,7,8])
│   ├── sum([1,2]) + sum([3,4])  │  sum([5,6]) + sum([7,8])
│   └── ...                       └── ...
```

## Testing
```bash
make
./test.sh
```
