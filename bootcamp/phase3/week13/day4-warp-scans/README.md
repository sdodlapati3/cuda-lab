# Day 4: Warp Scans

## Learning Objectives

- Implement inclusive and exclusive prefix scans
- Use __shfl_up_sync for scan operations
- Understand scan applications

## Key Concepts

### What is a Scan?

```
Input:   [1, 2, 3, 4, 5, 6, 7, 8]

Inclusive Scan (prefix sum):
         [1, 3, 6, 10, 15, 21, 28, 36]
         Each position = sum of all elements up to and including it

Exclusive Scan:
         [0, 1, 3, 6, 10, 15, 21, 28]
         Each position = sum of all elements BEFORE it
```

### Warp Scan Algorithm

Using `__shfl_up_sync`:

```
Step 1 (offset=1):  Each lane adds value from lane-1
Step 2 (offset=2):  Each lane adds value from lane-2
Step 3 (offset=4):  Each lane adds value from lane-4
Step 4 (offset=8):  Each lane adds value from lane-8
Step 5 (offset=16): Each lane adds value from lane-16

After 5 steps: inclusive scan complete for 32 elements!
```

### Scan Applications

- Stream compaction
- Radix sort
- Parallel allocation
- Polynomial evaluation
- Histogram equalization

## Build & Run

```bash
./build.sh
./build/warp_scan
```
