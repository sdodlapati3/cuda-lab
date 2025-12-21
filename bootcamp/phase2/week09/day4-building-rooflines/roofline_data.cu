/**
 * roofline_data.cu - Generate data points for roofline plot
 * 
 * Learning objectives:
 * - Measure ceilings and kernel performance
 * - Output data for plotting
 * - Understand where kernels fall on roofline
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Different kernels with varying arithmetic intensity

// AI ≈ 0.25 (1 FLOP per 4 bytes read)
__global__ void copy_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx];
}

// AI ≈ 0.125 (1 FLOP per 8 bytes: 4 read + 4 write)
__global__ void scale_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * 2.0f;
}

// AI ≈ 0.25 (2 FLOP per 8 bytes)
__global__ void saxpy_kernel(const float* x, const float* y, float* out, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a * x[idx] + y[idx];
}

// AI ≈ 2.5 (20 FLOPs per 8 bytes)
__global__ void high_ai_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        #pragma unroll
        for (int i = 0; i < 10; i++) {
            val = val * 1.001f + 0.001f;  // 2 FLOPs each
        }
        out[idx] = val;
    }
}

// AI ≈ 12.5 (100 FLOPs per 8 bytes)
__global__ void very_high_ai_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        #pragma unroll
        for (int i = 0; i < 50; i++) {
            val = val * 1.001f + 0.001f;
        }
        out[idx] = val;
    }
}

// AI ≈ 125 (1000 FLOPs per 8 bytes)
__global__ void extreme_ai_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        #pragma unroll
        for (int i = 0; i < 500; i++) {
            val = val * 1.001f + 0.001f;
        }
        out[idx] = val;
    }
}

struct KernelResult {
    const char* name;
    float ai;           // Arithmetic intensity
    float achieved_gflops;
    float achieved_bw;  // GB/s
};

int main() {
    const int N = 1 << 24;  // 16M elements
    const int TRIALS = 20;
    
    float *d_in, *d_out, *d_y;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    
    float* h_temp = new float[N];
    for (int i = 0; i < N; i++) h_temp[i] = 0.5f;
    cudaMemcpy(d_in, h_temp, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_temp, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    // Measure peak bandwidth first
    printf("=== Measuring Ceilings ===\n\n");
    
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        cudaMemcpy(d_out, d_in, N * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float peak_bw = 2.0f * N * sizeof(float) * TRIALS / (ms / 1000) / 1e9;
    printf("Peak Bandwidth: %.1f GB/s\n", peak_bw);
    
    // Get theoretical compute
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float gpu_clock_ghz = prop.clockRate / 1e6;
    int cuda_cores_per_sm = (prop.major == 8) ? ((prop.minor == 0) ? 64 : 128) : 64;
    float peak_flops = 2.0f * gpu_clock_ghz * prop.multiProcessorCount * cuda_cores_per_sm * 1000;
    printf("Peak Compute: %.1f GFLOPS\n", peak_flops);
    
    float ridge_point = peak_flops / peak_bw;
    printf("Ridge Point: %.1f FLOPS/byte\n", ridge_point);
    printf("\n");
    
    // Measure kernels
    KernelResult results[6];
    
    // Copy kernel (AI = 0.125 - just 0 FLOPs, 8 bytes)
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        copy_kernel<<<blocks, threads>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    results[0] = {"copy", 0.0f, 0.0f, 2.0f * N * sizeof(float) * TRIALS / (ms / 1000) / 1e9};
    
    // Scale kernel (AI = 0.125)
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        scale_kernel<<<blocks, threads>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float flops = (float)N * 1.0f;
    float bytes = 2.0f * N * sizeof(float);
    results[1] = {"scale", flops/bytes, flops * TRIALS / (ms/1000) / 1e9, bytes * TRIALS / (ms/1000) / 1e9};
    
    // SAXPY (AI = 0.25)
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        saxpy_kernel<<<blocks, threads>>>(d_in, d_y, d_out, 2.0f, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    flops = (float)N * 2.0f;  // 1 mul + 1 add
    bytes = 3.0f * N * sizeof(float);  // 2 reads + 1 write
    results[2] = {"saxpy", flops/bytes, flops * TRIALS / (ms/1000) / 1e9, bytes * TRIALS / (ms/1000) / 1e9};
    
    // High AI (AI = 2.5)
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        high_ai_kernel<<<blocks, threads>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    flops = (float)N * 20.0f;
    bytes = 2.0f * N * sizeof(float);
    results[3] = {"high_ai(20)", flops/bytes, flops * TRIALS / (ms/1000) / 1e9, bytes * TRIALS / (ms/1000) / 1e9};
    
    // Very high AI (AI = 12.5)
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        very_high_ai_kernel<<<blocks, threads>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    flops = (float)N * 100.0f;
    bytes = 2.0f * N * sizeof(float);
    results[4] = {"very_high_ai(100)", flops/bytes, flops * TRIALS / (ms/1000) / 1e9, bytes * TRIALS / (ms/1000) / 1e9};
    
    // Extreme AI (AI = 125)
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        extreme_ai_kernel<<<blocks, threads>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    flops = (float)N * 1000.0f;
    bytes = 2.0f * N * sizeof(float);
    results[5] = {"extreme_ai(1000)", flops/bytes, flops * TRIALS / (ms/1000) / 1e9, bytes * TRIALS / (ms/1000) / 1e9};
    
    // Print results
    printf("=== Kernel Performance ===\n\n");
    printf("%-20s %10s %12s %10s %s\n", "Kernel", "AI", "GFLOPS", "GB/s", "Bound");
    printf("------------------------------------------------------------------------\n");
    
    for (int i = 0; i < 6; i++) {
        const char* bound = (results[i].ai < ridge_point) ? "memory" : "compute";
        printf("%-20s %10.2f %12.1f %10.1f %s\n", 
               results[i].name, results[i].ai, results[i].achieved_gflops, 
               results[i].achieved_bw, bound);
    }
    
    // Output CSV for plotting
    printf("\n=== Data for Plotting (roofline_data.csv) ===\n\n");
    
    FILE* f = fopen("roofline_data.csv", "w");
    fprintf(f, "kernel,ai,gflops\n");
    fprintf(f, "peak_bw,%.2f,%.1f\n", ridge_point, peak_flops);  // Ridge point
    for (int i = 0; i < 6; i++) {
        fprintf(f, "%s,%.2f,%.1f\n", results[i].name, results[i].ai, results[i].achieved_gflops);
    }
    fclose(f);
    
    printf("Data written to roofline_data.csv\n");
    printf("Use plot_roofline.py to visualize\n");
    
    // Print roofline data points
    printf("\n=== Roofline Model Data ===\n");
    printf("Peak BW ceiling: %.1f GB/s\n", peak_bw);
    printf("Peak compute ceiling: %.1f GFLOPS\n", peak_flops);
    printf("Ridge point AI: %.1f FLOPS/byte\n", ridge_point);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_y);
    delete[] h_temp;
    
    return 0;
}
