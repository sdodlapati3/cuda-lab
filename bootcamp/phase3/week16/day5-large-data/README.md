# Day 5: Large Data Handling

## Learning Objectives

- Process data larger than GPU memory
- Implement chunked streaming
- Design out-of-core algorithms

## Key Concepts

### The Problem

```
GPU Memory: 16 GB
Your Data:  100 GB

Solution: Process in chunks
```

### Streaming Pattern

```cpp
const int CHUNK_SIZE = 1 << 30;  // 1 GB chunks
for (offset = 0; offset < total_size; offset += CHUNK_SIZE) {
    // Load chunk
    cudaMemcpyAsync(d_chunk, h_data + offset, CHUNK_SIZE, H2D, stream);
    
    // Process
    kernel<<<..., stream>>>(d_chunk);
    
    // Store results
    cudaMemcpyAsync(h_results + offset, d_chunk, CHUNK_SIZE, D2H, stream);
}
```

### Double Buffering

```cpp
// While chunk N processes, load chunk N+1
while (hasMoreChunks()) {
    cudaMemcpyAsync(d_buf[next], h_chunk[next], size, H2D, loadStream);
    kernel<<<..., computeStream>>>(d_buf[curr]);
    cudaMemcpyAsync(h_out[prev], d_buf[prev], size, D2H, storeStream);
    
    swap buffers;
}
```

### Memory-Mapped Files

```cpp
// Map huge file to virtual address
int fd = open("huge_file.bin", O_RDONLY);
void* mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

// Process chunks
for (chunk in chunks) {
    cudaMemcpy(d_data, mapped + offset, chunk_size, H2D);
    process(d_data);
}
```

## Build & Run

```bash
./build.sh
./build/large_data
```
