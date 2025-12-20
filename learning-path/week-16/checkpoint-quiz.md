# Week 16 Checkpoint Quiz: VMM & Advanced Memory

## Instructions
- Answer all questions
- 30 points total
- Passing score: 24/30

---

## Section A: Conceptual Questions (12 points)

### Q1 (3 points)
What are the four main steps in the VMM workflow for allocating GPU memory?

### Q2 (3 points)
What is the advantage of `cudaMallocAsync` over `cudaMalloc` in a stream-based workflow?

### Q3 (3 points)
How does a memory pool's "release threshold" affect memory usage?

### Q4 (3 points)
Why is VMM useful for building growable data structures on the GPU?

---

## Section B: Code Analysis (10 points)

### Q5 (5 points)
What does this code do and what problem does it solve?

```cpp
CUmemAccessDesc accessDescs[2];
for (int i = 0; i < 2; i++) {
    accessDescs[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDescs[i].location.id = i;
    accessDescs[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
}
cuMemSetAccess(ptr, size, accessDescs, 2);
```

### Q6 (5 points)
What is wrong with this code and how would you fix it?

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

float* data;
cudaMallocAsync(&data, 1024, stream1);
kernel<<<1, 256, 0, stream2>>>(data);  // Bug!
cudaFreeAsync(data, stream2);
```

---

## Section C: Practical Application (8 points)

### Q7 (4 points)
You need to implement a GPU buffer that:
- Starts at 1 MB
- Can grow to 1 GB
- Should not copy data when growing

Which approach would you use and why?
a) `cudaMalloc` + `cudaMemcpy` when growing
b) `cudaMallocManaged` with overcommit
c) VMM with `cuMemAddressReserve` and `cuMemMap`

### Q8 (4 points)
In a multi-GPU training system, you want GPU 1 to directly access memory on GPU 0 without explicit copies. Describe the VMM API calls needed to set this up.

---

## Answer Key

### Q1
1. `cuMemAddressReserve` - Reserve virtual address range
2. `cuMemCreate` - Create physical memory handle
3. `cuMemMap` - Map physical to virtual
4. `cuMemSetAccess` - Set read/write permissions

### Q2
`cudaMallocAsync` is non-blocking and tied to stream ordering. It doesn't implicitly synchronize the device, allowing the async execution pipeline to continue without stalls. `cudaMalloc` synchronizes, breaking async execution.

### Q3
The release threshold controls when freed memory is returned to the system. Memory below the threshold is kept in the pool for fast reuse. Memory above the threshold may be released. Higher threshold = more memory reserved for reuse but higher memory footprint.

### Q4
VMM allows separating virtual address reservation (cheap, can be huge) from physical allocation (expensive, actual memory). You can reserve a large VA range, then map physical memory incrementally. When growing, you just map more physical memory to the already-reserved VA - no need to copy existing data.

### Q5
This code grants read/write access to a VMM allocation for both GPU 0 and GPU 1. It enables peer-to-peer memory access where either GPU can read from or write to the memory. This solves the problem of multi-GPU memory sharing with fine-grained access control.

### Q6
Bug: Memory is allocated on `stream1` but used on `stream2` without synchronization. `stream2` may try to use the memory before `stream1`'s allocation completes.

Fix: Add an event to synchronize:
```cpp
cudaEvent_t event;
cudaEventCreate(&event);
cudaMallocAsync(&data, 1024, stream1);
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event);  // stream2 waits for stream1
kernel<<<1, 256, 0, stream2>>>(data);
cudaFreeAsync(data, stream2);
```

### Q7
Answer: c) VMM with `cuMemAddressReserve` and `cuMemMap`

Reasoning:
- Option (a) requires copying all data when growing - O(n) cost
- Option (b) has implicit migration overhead and less control
- Option (c) reserves 1GB VA upfront, maps 1MB physical initially, can add more physical memory without copying because the virtual addresses don't change

### Q8
```cpp
// 1. On GPU 0, create allocation with VMM
CUdeviceptr ptr;
cuMemAddressReserve(&ptr, size, granularity, 0, 0);
cuMemCreate(&handle, size, &prop, 0);  // prop.location.id = 0
cuMemMap(ptr, size, 0, handle, 0);

// 2. Grant access to both GPU 0 and GPU 1
CUmemAccessDesc accessDescs[2];
accessDescs[0].location.id = 0;
accessDescs[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
accessDescs[1].location.id = 1;
accessDescs[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
cuMemSetAccess(ptr, size, accessDescs, 2);

// Now GPU 1 can directly access ptr
```
