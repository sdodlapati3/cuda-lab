# Week 17 Checkpoint Quiz: IPC & Textures

## Section A: Inter-Process Communication (5 questions)

### Q1: IPC Memory Handles
Which function exports a GPU memory handle for sharing between processes?
- A) `cudaMemGetInfo()`
- B) `cudaIpcGetMemHandle()`
- C) `cudaMallocHost()`
- D) `cudaHostRegister()`

<details>
<summary>Answer</summary>
B) `cudaIpcGetMemHandle()` - Creates a memory handle that can be transferred to another process.
</details>

---

### Q2: IPC Requirements
What is a requirement for CUDA IPC to work?
- A) Processes must be on different machines
- B) Processes must run on the same GPU device
- C) Processes must use MPI
- D) Processes must share the same executable

<details>
<summary>Answer</summary>
B) Processes must run on the same GPU device. IPC allows sharing memory between processes on the same node using the same GPU.
</details>

---

### Q3: IPC Handle Transfer
How are IPC memory handles typically transferred between processes?
- A) Through CUDA unified memory
- B) Through shared files, pipes, or sockets
- C) Automatically by the driver
- D) Through PCIe directly

<details>
<summary>Answer</summary>
B) Through shared files, pipes, or sockets. The `cudaIpcMemHandle_t` is a 64-byte opaque struct that must be sent via standard IPC mechanisms.
</details>

---

### Q4: Opening IPC Memory
What does `cudaIpcOpenMemHandle()` return?
- A) A file descriptor
- B) A device pointer to the shared memory
- C) The size of the allocation
- D) A stream object

<details>
<summary>Answer</summary>
B) A device pointer to the shared memory. The consumer process uses this to access memory allocated by the producer.
</details>

---

### Q5: IPC Cleanup
What happens if you call `cudaFree()` on memory opened via IPC?
- A) The memory is freed normally
- B) Undefined behavior / error
- C) Only the mapping is released
- D) Other processes lose access

<details>
<summary>Answer</summary>
B) Undefined behavior / error. You must use `cudaIpcCloseMemHandle()` to release IPC mappings, not `cudaFree()`.
</details>

---

## Section B: Texture Objects (5 questions)

### Q6: Texture Object Creation
Which structure describes how to sample a texture (addressing, filtering)?
- A) `cudaResourceDesc`
- B) `cudaTextureDesc`
- C) `cudaChannelFormatDesc`
- D) `cudaMemcpyKind`

<details>
<summary>Answer</summary>
B) `cudaTextureDesc` - Contains addressing mode, filter mode, normalized coordinates settings.
</details>

---

### Q7: Filter Modes
What does `cudaFilterModeLinear` provide for 2D textures?
- A) Nearest-neighbor sampling
- B) Bilinear interpolation
- C) Cubic interpolation
- D) Anisotropic filtering

<details>
<summary>Answer</summary>
B) Bilinear interpolation. The hardware automatically interpolates between the four nearest texels.
</details>

---

### Q8: Address Modes
If texture coordinates are outside [0, 1] with `cudaAddressModeWrap`:
- A) Coordinates are clamped to [0, 1]
- B) Coordinates wrap around (modulo)
- C) Zero is returned
- D) An error occurs

<details>
<summary>Answer</summary>
B) Coordinates wrap around (modulo). This is useful for seamlessly tiling textures.
</details>

---

### Q9: CUDA Arrays
Why are CUDA arrays preferred for 2D textures?
- A) They use less memory
- B) They support better 2D spatial locality caching
- C) They are faster to allocate
- D) They work on all GPU architectures

<details>
<summary>Answer</summary>
B) They support better 2D spatial locality caching. CUDA arrays use optimized memory layouts for texture cache.
</details>

---

### Q10: Normalized vs Unnormalized Coordinates
With normalized coordinates, texture coordinate (0.5, 0.5) in a 100x100 texture refers to:
- A) Pixel (0, 0)
- B) Pixel (50, 50)
- C) Pixel (100, 100)
- D) The center of the texture

<details>
<summary>Answer</summary>
D) The center of the texture. Normalized coordinates map [0, 1] to the entire texture, so (0.5, 0.5) is the center.
</details>

---

## Section C: Practical Applications (5 questions)

### Q11: LUT with Textures
Why use texture memory for lookup tables?
- A) Larger capacity than global memory
- B) Free hardware interpolation + cache
- C) Simpler programming model
- D) Lower latency than registers

<details>
<summary>Answer</summary>
B) Free hardware interpolation + cache. Texture units provide L1 cache and automatic interpolation at no extra cost.
</details>

---

### Q12: Image Convolution Benefit
What texture feature simplifies image convolution?
- A) Automatic padding with zeros
- B) Address modes handle border pixels automatically
- C) Built-in convolution kernels
- D) Higher precision math

<details>
<summary>Answer</summary>
B) Address modes handle border pixels automatically. Clamp mode returns edge values for out-of-bounds access.
</details>

---

### Q13: IPC Ring Buffer
In an IPC ring buffer pattern, what prevents race conditions?
- A) CUDA streams
- B) File locks and atomic operations
- C) Unified memory
- D) Multiple GPU contexts

<details>
<summary>Answer</summary>
B) File locks and atomic operations. Producer-consumer synchronization requires explicit coordination.
</details>

---

### Q14: Multi-GPU IPC
What additional step is needed for IPC between different GPUs on the same node?
- A) Use NCCL instead
- B) Enable peer access with `cudaDeviceEnablePeerAccess()`
- C) Use managed memory
- D) IPC between GPUs is not supported

<details>
<summary>Answer</summary>
B) Enable peer access with `cudaDeviceEnablePeerAccess()`. This allows direct P2P memory access between GPUs.
</details>

---

### Q15: Texture vs Global Memory
When does texture memory provide the most benefit?
- A) Sequential access patterns
- B) Random 2D spatial access with reuse
- C) Write-heavy workloads
- D) Very large allocations

<details>
<summary>Answer</summary>
B) Random 2D spatial access with reuse. Texture cache is optimized for 2D locality and repeated sampling.
</details>

---

## Scoring

| Score | Level |
|-------|-------|
| 13-15 | Expert - Ready for advanced topics |
| 10-12 | Proficient - Solid understanding |
| 7-9 | Developing - Review weak areas |
| < 7 | Beginner - Re-study materials |

## Key Takeaways

1. **IPC enables zero-copy sharing** between processes on the same GPU
2. **Texture objects** replace legacy texture references (deprecated)
3. **Filter modes** provide free interpolation in hardware
4. **Address modes** simplify boundary handling
5. **CUDA arrays** optimize 2D/3D data for texture cache
