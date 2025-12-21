/**
 * thrust_demo.cu - Thrust STL-like GPU programming
 * 
 * Learning objectives:
 * - Use device_vector for automatic memory
 * - Apply transformations and reductions
 * - Interop with raw CUDA pointers
 */

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/generate.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <random>

// ============================================================================
// Custom Functors
// ============================================================================

struct square {
    __host__ __device__
    float operator()(float x) const { return x * x; }
};

struct saxpy_functor {
    const float a;
    saxpy_functor(float _a) : a(_a) {}
    
    __host__ __device__
    float operator()(float x, float y) const {
        return a * x + y;
    }
};

// ============================================================================
// Raw CUDA kernel for interop demo
// ============================================================================

__global__ void my_kernel(float* data, int n, float scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scalar;
    }
}

int main() {
    printf("=== Thrust Library Demo ===\n\n");
    
    const int N = 1 << 20;  // 1M elements
    
    // ========================================================================
    // Part 1: Basic device_vector usage
    // ========================================================================
    {
        printf("1. device_vector Basics\n");
        printf("─────────────────────────────────────────\n");
        
        // Automatic GPU allocation
        thrust::device_vector<float> d_vec(N);
        
        // Fill with value
        thrust::fill(d_vec.begin(), d_vec.end(), 1.0f);
        
        // Reduce
        float sum = thrust::reduce(d_vec.begin(), d_vec.end());
        
        printf("   Vector of %d ones\n", N);
        printf("   Sum: %.0f (expected %d)\n\n", sum, N);
    }
    
    // ========================================================================
    // Part 2: Transform Operations
    // ========================================================================
    {
        printf("2. Transform Operations\n");
        printf("─────────────────────────────────────────\n");
        
        thrust::device_vector<float> d_x(N);
        thrust::device_vector<float> d_y(N);
        thrust::device_vector<float> d_result(N);
        
        // Initialize
        thrust::fill(d_x.begin(), d_x.end(), 2.0f);
        thrust::fill(d_y.begin(), d_y.end(), 3.0f);
        
        // Square transform: result[i] = x[i]^2
        thrust::transform(d_x.begin(), d_x.end(), d_result.begin(), square());
        
        float first_squared = d_result[0];
        printf("   square(2.0) = %.1f\n", first_squared);
        
        // SAXPY: result[i] = a * x[i] + y[i]
        float a = 2.0f;
        thrust::transform(d_x.begin(), d_x.end(),
                          d_y.begin(), d_result.begin(),
                          saxpy_functor(a));
        
        float first_saxpy = d_result[0];
        printf("   SAXPY(2.0*2.0 + 3.0) = %.1f\n\n", first_saxpy);
    }
    
    // ========================================================================
    // Part 3: Sorting
    // ========================================================================
    {
        printf("3. Sorting\n");
        printf("─────────────────────────────────────────\n");
        
        thrust::device_vector<int> d_keys(N);
        
        // Generate random keys
        thrust::host_vector<int> h_keys(N);
        std::mt19937 rng(42);
        for (int i = 0; i < N; i++) h_keys[i] = rng() % 1000;
        
        d_keys = h_keys;  // Copy to device
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        thrust::sort(d_keys.begin(), d_keys.end());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        
        // Verify sorted
        thrust::host_vector<int> h_sorted = d_keys;
        bool sorted = true;
        for (int i = 1; i < N && sorted; i++) {
            if (h_sorted[i] < h_sorted[i-1]) sorted = false;
        }
        
        printf("   Sorted %d integers in %.2f ms\n", N, ms);
        printf("   Verification: %s\n", sorted ? "PASSED" : "FAILED");
        printf("   First 5: %d %d %d %d %d\n", 
               h_sorted[0], h_sorted[1], h_sorted[2], h_sorted[3], h_sorted[4]);
        printf("   Last 5:  %d %d %d %d %d\n\n",
               h_sorted[N-5], h_sorted[N-4], h_sorted[N-3], h_sorted[N-2], h_sorted[N-1]);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // ========================================================================
    // Part 4: Sort by Key
    // ========================================================================
    {
        printf("4. Sort by Key (Key-Value Pairs)\n");
        printf("─────────────────────────────────────────\n");
        
        const int n = 8;
        thrust::device_vector<int> keys(n);
        thrust::device_vector<float> values(n);
        
        // Create unsorted key-value pairs
        thrust::host_vector<int> h_keys = {3, 1, 4, 1, 5, 9, 2, 6};
        thrust::host_vector<float> h_values = {30.f, 10.f, 40.f, 11.f, 50.f, 90.f, 20.f, 60.f};
        
        keys = h_keys;
        values = h_values;
        
        thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
        
        thrust::host_vector<int> sorted_keys = keys;
        thrust::host_vector<float> sorted_values = values;
        
        printf("   Original: keys=[3,1,4,1,5,9,2,6]\n");
        printf("   Sorted:   keys=[");
        for (int i = 0; i < n; i++) printf("%d%s", sorted_keys[i], i<n-1?",":"");
        printf("]\n");
        printf("             vals=[");
        for (int i = 0; i < n; i++) printf("%.0f%s", sorted_values[i], i<n-1?",":"");
        printf("]\n\n");
    }
    
    // ========================================================================
    // Part 5: Reductions
    // ========================================================================
    {
        printf("5. Reduction Operations\n");
        printf("─────────────────────────────────────────\n");
        
        thrust::device_vector<float> d_vec(N);
        thrust::sequence(d_vec.begin(), d_vec.end(), 1.0f);  // 1, 2, 3, ...
        
        // Sum
        float sum = thrust::reduce(d_vec.begin(), d_vec.end());
        
        // Min/Max
        float min_val = *thrust::min_element(d_vec.begin(), d_vec.end());
        float max_val = *thrust::max_element(d_vec.begin(), d_vec.end());
        
        // Count
        int count = thrust::count_if(d_vec.begin(), d_vec.end(),
            [] __device__ (float x) { return x > N/2; });
        
        printf("   sequence 1 to %d\n", N);
        printf("   Sum: %.0f\n", sum);
        printf("   Min: %.0f, Max: %.0f\n", min_val, max_val);
        printf("   Count > %d: %d\n\n", N/2, count);
    }
    
    // ========================================================================
    // Part 6: Interop with Raw CUDA
    // ========================================================================
    {
        printf("6. Raw CUDA Interop\n");
        printf("─────────────────────────────────────────\n");
        
        thrust::device_vector<float> d_vec(1000);
        thrust::fill(d_vec.begin(), d_vec.end(), 2.0f);
        
        // Get raw pointer
        float* raw_ptr = thrust::raw_pointer_cast(d_vec.data());
        
        // Use in CUDA kernel
        int block_size = 256;
        int num_blocks = (1000 + block_size - 1) / block_size;
        my_kernel<<<num_blocks, block_size>>>(raw_ptr, 1000, 5.0f);
        cudaDeviceSynchronize();
        
        float first = d_vec[0];
        printf("   Started with 2.0, multiplied by 5.0 in kernel\n");
        printf("   Result: %.1f\n\n", first);
        
        // Wrap existing CUDA memory
        float* d_cuda_ptr;
        cudaMalloc(&d_cuda_ptr, 100 * sizeof(float));
        
        thrust::device_ptr<float> d_thrust_ptr(d_cuda_ptr);
        thrust::fill(d_thrust_ptr, d_thrust_ptr + 100, 42.0f);
        
        float check;
        cudaMemcpy(&check, d_cuda_ptr, sizeof(float), cudaMemcpyDeviceToHost);
        printf("   Wrapped raw pointer, filled with thrust\n");
        printf("   Value: %.0f\n\n", check);
        
        cudaFree(d_cuda_ptr);
    }
    
    // ========================================================================
    // Part 7: Zip Iterators
    // ========================================================================
    {
        printf("7. Zip Iterators\n");
        printf("─────────────────────────────────────────\n");
        
        thrust::device_vector<float> d_x(5);
        thrust::device_vector<float> d_y(5);
        thrust::device_vector<float> d_z(5);
        
        thrust::sequence(d_x.begin(), d_x.end(), 1.0f);  // 1,2,3,4,5
        thrust::sequence(d_y.begin(), d_y.end(), 10.0f); // 10,11,12,13,14
        
        // Transform zip of (x,y) -> z = x + y
        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(d_x.begin(), d_y.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(d_x.end(), d_y.end())),
            d_z.begin(),
            [] __device__ (thrust::tuple<float, float> t) {
                return thrust::get<0>(t) + thrust::get<1>(t);
            }
        );
        
        thrust::host_vector<float> h_z = d_z;
        printf("   x = [1,2,3,4,5], y = [10,11,12,13,14]\n");
        printf("   z = x + y = [");
        for (int i = 0; i < 5; i++) printf("%.0f%s", h_z[i], i<4?",":"");
        printf("]\n\n");
    }
    
    printf("=== Key Points ===\n\n");
    printf("1. device_vector: automatic GPU memory management\n");
    printf("2. STL-like algorithms: transform, reduce, sort\n");
    printf("3. Custom functors for operations\n");
    printf("4. raw_pointer_cast for CUDA interop\n");
    printf("5. device_ptr wraps existing CUDA memory\n");
    
    return 0;
}
