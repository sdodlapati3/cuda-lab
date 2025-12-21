/**
 * large_data.cu - Processing data larger than GPU memory
 * 
 * Learning objectives:
 * - Chunked processing
 * - Double buffering
 * - Overlap transfer with compute
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

// Compute-intensive kernel
__global__ void process_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < 50; i++) {
            val = sinf(val) * cosf(val) + 1.0f;
        }
        data[idx] = val;
    }
}

int main() {
    printf("=== Large Data Handling Demo ===\n\n");
    
    // Simulate "large" data (in reality would be much larger)
    const size_t TOTAL_SIZE = 256 << 20;  // 256 MB "large" data
    const size_t CHUNK_SIZE = 64 << 20;   // 64 MB chunks
    const int NUM_CHUNKS = TOTAL_SIZE / CHUNK_SIZE;
    const int ELEMENTS_PER_CHUNK = CHUNK_SIZE / sizeof(float);
    
    printf("Total data: %zu MB\n", TOTAL_SIZE >> 20);
    printf("Chunk size: %zu MB\n", CHUNK_SIZE >> 20);
    printf("Num chunks: %d\n\n", NUM_CHUNKS);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ========================================================================
    // Part 1: Simple Sequential Processing
    // ========================================================================
    {
        printf("1. Sequential Chunk Processing\n");
        printf("─────────────────────────────────────────\n");
        
        // Allocate pinned host memory for all data
        float* h_data;
        cudaMallocHost(&h_data, TOTAL_SIZE);
        for (size_t i = 0; i < TOTAL_SIZE / sizeof(float); i++) {
            h_data[i] = 1.0f;
        }
        
        // Single device buffer
        float* d_buffer;
        cudaMalloc(&d_buffer, CHUNK_SIZE);
        
        cudaEventRecord(start);
        
        for (int c = 0; c < NUM_CHUNKS; c++) {
            size_t offset = c * ELEMENTS_PER_CHUNK;
            
            // H2D
            cudaMemcpy(d_buffer, h_data + offset, CHUNK_SIZE, cudaMemcpyHostToDevice);
            
            // Process
            process_kernel<<<(ELEMENTS_PER_CHUNK + 255) / 256, 256>>>(
                d_buffer, ELEMENTS_PER_CHUNK);
            
            // D2H
            cudaMemcpy(h_data + offset, d_buffer, CHUNK_SIZE, cudaMemcpyDeviceToHost);
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float seq_ms;
        cudaEventElapsedTime(&seq_ms, start, stop);
        
        printf("   Sequential: %.2f ms\n", seq_ms);
        
        cudaFree(d_buffer);
        cudaFreeHost(h_data);
    }
    
    // ========================================================================
    // Part 2: Async with Single Stream
    // ========================================================================
    {
        printf("\n2. Async Single Stream\n");
        printf("─────────────────────────────────────────\n");
        
        float* h_data;
        cudaMallocHost(&h_data, TOTAL_SIZE);
        for (size_t i = 0; i < TOTAL_SIZE / sizeof(float); i++) {
            h_data[i] = 1.0f;
        }
        
        float* d_buffer;
        cudaMalloc(&d_buffer, CHUNK_SIZE);
        
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        cudaEventRecord(start);
        
        for (int c = 0; c < NUM_CHUNKS; c++) {
            size_t offset = c * ELEMENTS_PER_CHUNK;
            
            cudaMemcpyAsync(d_buffer, h_data + offset, CHUNK_SIZE,
                           cudaMemcpyHostToDevice, stream);
            process_kernel<<<(ELEMENTS_PER_CHUNK + 255) / 256, 256, 0, stream>>>(
                d_buffer, ELEMENTS_PER_CHUNK);
            cudaMemcpyAsync(h_data + offset, d_buffer, CHUNK_SIZE,
                           cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float async_ms;
        cudaEventElapsedTime(&async_ms, start, stop);
        
        printf("   Async single stream: %.2f ms\n", async_ms);
        
        cudaStreamDestroy(stream);
        cudaFree(d_buffer);
        cudaFreeHost(h_data);
    }
    
    // ========================================================================
    // Part 3: Double Buffering
    // ========================================================================
    {
        printf("\n3. Double Buffering (Overlap Transfer + Compute)\n");
        printf("─────────────────────────────────────────\n");
        
        float* h_data;
        cudaMallocHost(&h_data, TOTAL_SIZE);
        for (size_t i = 0; i < TOTAL_SIZE / sizeof(float); i++) {
            h_data[i] = 1.0f;
        }
        
        // Double buffer on device
        float* d_buffer[2];
        cudaMalloc(&d_buffer[0], CHUNK_SIZE);
        cudaMalloc(&d_buffer[1], CHUNK_SIZE);
        
        // Separate streams
        cudaStream_t streams[2];
        cudaStreamCreate(&streams[0]);
        cudaStreamCreate(&streams[1]);
        
        cudaEventRecord(start);
        
        // Load first chunk
        cudaMemcpyAsync(d_buffer[0], h_data, CHUNK_SIZE,
                       cudaMemcpyHostToDevice, streams[0]);
        
        for (int c = 0; c < NUM_CHUNKS; c++) {
            int curr = c % 2;
            int next = (c + 1) % 2;
            size_t curr_offset = c * ELEMENTS_PER_CHUNK;
            size_t next_offset = (c + 1) * ELEMENTS_PER_CHUNK;
            
            // Start loading next chunk while processing current
            if (c + 1 < NUM_CHUNKS) {
                cudaMemcpyAsync(d_buffer[next], h_data + next_offset, CHUNK_SIZE,
                               cudaMemcpyHostToDevice, streams[next]);
            }
            
            // Process current chunk
            process_kernel<<<(ELEMENTS_PER_CHUNK + 255) / 256, 256, 0, streams[curr]>>>(
                d_buffer[curr], ELEMENTS_PER_CHUNK);
            
            // Store current chunk
            cudaMemcpyAsync(h_data + curr_offset, d_buffer[curr], CHUNK_SIZE,
                           cudaMemcpyDeviceToHost, streams[curr]);
        }
        
        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float double_ms;
        cudaEventElapsedTime(&double_ms, start, stop);
        
        printf("   Double buffering: %.2f ms\n", double_ms);
        
        cudaStreamDestroy(streams[0]);
        cudaStreamDestroy(streams[1]);
        cudaFree(d_buffer[0]);
        cudaFree(d_buffer[1]);
        cudaFreeHost(h_data);
    }
    
    // ========================================================================
    // Part 4: Triple Buffering (Load + Compute + Store)
    // ========================================================================
    {
        printf("\n4. Triple Buffering (Full Pipeline)\n");
        printf("─────────────────────────────────────────\n");
        
        float* h_data;
        cudaMallocHost(&h_data, TOTAL_SIZE);
        for (size_t i = 0; i < TOTAL_SIZE / sizeof(float); i++) {
            h_data[i] = 1.0f;
        }
        
        // Triple buffer
        float* d_buffer[3];
        for (int i = 0; i < 3; i++) {
            cudaMalloc(&d_buffer[i], CHUNK_SIZE);
        }
        
        cudaStream_t loadStream, computeStream, storeStream;
        cudaStreamCreate(&loadStream);
        cudaStreamCreate(&computeStream);
        cudaStreamCreate(&storeStream);
        
        cudaEvent_t loadDone[3], computeDone[3];
        for (int i = 0; i < 3; i++) {
            cudaEventCreate(&loadDone[i]);
            cudaEventCreate(&computeDone[i]);
        }
        
        cudaEventRecord(start);
        
        for (int c = 0; c < NUM_CHUNKS + 2; c++) {
            int load_idx = c % 3;
            int compute_idx = (c - 1 + 3) % 3;
            int store_idx = (c - 2 + 3) % 3;
            
            // Load phase
            if (c < NUM_CHUNKS) {
                if (c > 0) {
                    cudaStreamWaitEvent(loadStream, computeDone[load_idx]);
                }
                cudaMemcpyAsync(d_buffer[load_idx], 
                               h_data + c * ELEMENTS_PER_CHUNK, CHUNK_SIZE,
                               cudaMemcpyHostToDevice, loadStream);
                cudaEventRecord(loadDone[load_idx], loadStream);
            }
            
            // Compute phase
            if (c > 0 && c <= NUM_CHUNKS) {
                cudaStreamWaitEvent(computeStream, loadDone[compute_idx]);
                process_kernel<<<(ELEMENTS_PER_CHUNK + 255) / 256, 256, 0, computeStream>>>(
                    d_buffer[compute_idx], ELEMENTS_PER_CHUNK);
                cudaEventRecord(computeDone[compute_idx], computeStream);
            }
            
            // Store phase
            if (c > 1) {
                int store_chunk = c - 2;
                if (store_chunk < NUM_CHUNKS) {
                    cudaStreamWaitEvent(storeStream, computeDone[store_idx]);
                    cudaMemcpyAsync(h_data + store_chunk * ELEMENTS_PER_CHUNK,
                                   d_buffer[store_idx], CHUNK_SIZE,
                                   cudaMemcpyDeviceToHost, storeStream);
                }
            }
        }
        
        cudaStreamSynchronize(storeStream);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float triple_ms;
        cudaEventElapsedTime(&triple_ms, start, stop);
        
        printf("   Triple buffering: %.2f ms\n", triple_ms);
        
        // Cleanup
        cudaStreamDestroy(loadStream);
        cudaStreamDestroy(computeStream);
        cudaStreamDestroy(storeStream);
        for (int i = 0; i < 3; i++) {
            cudaFree(d_buffer[i]);
            cudaEventDestroy(loadDone[i]);
            cudaEventDestroy(computeDone[i]);
        }
        cudaFreeHost(h_data);
    }
    
    printf("\n=== Key Points ===\n\n");
    printf("1. Chunk data to fit in GPU memory\n");
    printf("2. Use pinned memory for async transfers\n");
    printf("3. Double buffer overlaps load+compute or compute+store\n");
    printf("4. Triple buffer overlaps all three phases\n");
    printf("5. Choose chunk size based on compute/transfer ratio\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
