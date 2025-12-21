/**
 * spinlock_demo.cu - Spinlocks on GPU (and why to avoid them)
 * 
 * Learning objectives:
 * - Implement spinlock
 * - Understand dangers
 * - See performance impact
 */

#include <cuda_runtime.h>
#include <cstdio>

// Simple spinlock
class SpinLock {
public:
    __device__ void lock(int* mutex) {
        while (atomicCAS(mutex, 0, 1) != 0) {
            // Spin
        }
    }
    
    __device__ void unlock(int* mutex) {
        atomicExch(mutex, 0);
    }
};

// Spinlock with backoff (better for contention)
class BackoffSpinLock {
public:
    __device__ void lock(int* mutex) {
        int backoff = 1;
        while (atomicCAS(mutex, 0, 1) != 0) {
            // Exponential backoff
            for (int i = 0; i < backoff; i++) {
                __nanosleep(32);
            }
            backoff = min(backoff * 2, 256);
        }
    }
    
    __device__ void unlock(int* mutex) {
        atomicExch(mutex, 0);
    }
};

// DANGEROUS: Can deadlock due to warp scheduling
__global__ void spinlock_increment(int* mutex, int* counter, int iterations) {
    SpinLock lock;
    
    for (int i = 0; i < iterations; i++) {
        lock.lock(mutex);
        (*counter)++;
        lock.unlock(mutex);
    }
}

// Safer: One thread per warp holds lock
__global__ void warp_spinlock_increment(int* mutex, int* counter, int iterations) {
    SpinLock lock;
    int lane = threadIdx.x % 32;
    
    for (int i = 0; i < iterations; i++) {
        // Only lane 0 of each warp participates
        if (lane == 0) {
            lock.lock(mutex);
            (*counter)++;
            lock.unlock(mutex);
        }
        __syncwarp();
    }
}

// Much better: Just use atomics!
__global__ void atomic_increment(int* counter, int iterations) {
    for (int i = 0; i < iterations; i++) {
        atomicAdd(counter, 1);
    }
}

// Critical section example: updating a struct
struct Account {
    int balance;
    int transactions;
};

__global__ void spinlock_transfer(int* mutex, Account* account, int amount, int count) {
    SpinLock lock;
    int lane = threadIdx.x % 32;
    
    for (int i = 0; i < count; i++) {
        if (lane == 0) {
            lock.lock(mutex);
            // Critical section - multiple operations
            account->balance += amount;
            account->transactions++;
            lock.unlock(mutex);
        }
        __syncwarp();
    }
}

// Lock-free alternative using packed atomics
__global__ void lockfree_transfer(unsigned long long* packed_account, int amount, int count) {
    for (int i = 0; i < count; i++) {
        unsigned long long old_val, new_val;
        do {
            old_val = *packed_account;
            int balance = (int)(old_val >> 32);
            int transactions = (int)(old_val & 0xFFFFFFFF);
            balance += amount;
            transactions++;
            new_val = ((unsigned long long)balance << 32) | (unsigned)transactions;
        } while (atomicCAS(packed_account, old_val, new_val) != old_val);
    }
}

__global__ void demonstrate_deadlock_danger() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("=== Spinlock Deadlock Dangers ===\n\n");
        
        printf("Scenario: Warp with threads 0-31\n");
        printf("  Thread 0 acquires lock\n");
        printf("  Threads 1-31 spin waiting for lock\n");
        printf("  Problem: Thread 0 can't proceed because\n");
        printf("           warp executes in lockstep!\n\n");
        
        printf("Solutions:\n");
        printf("1. Only one thread per warp acquires lock\n");
        printf("2. Use lock-free algorithms instead\n");
        printf("3. Use cooperative groups for fine control\n");
        printf("4. Best: avoid locks on GPU entirely!\n");
    }
}

int main() {
    printf("=== Spinlock Demo ===\n\n");
    
    int* d_mutex;
    int* d_counter;
    cudaMalloc(&d_mutex, sizeof(int));
    cudaMalloc(&d_counter, sizeof(int));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    const int ITERATIONS = 1000;
    const int BLOCKS = 8;
    const int THREADS = 32;  // One warp per block for safety
    int expected = BLOCKS * THREADS * ITERATIONS;
    
    printf("Config: %d blocks x %d threads x %d iterations\n", BLOCKS, THREADS, ITERATIONS);
    printf("Expected count: %d\n\n", expected);
    
    // Warp-safe spinlock (only lane 0)
    int zero = 0;
    cudaMemcpy(d_mutex, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counter, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    int expected_warp = BLOCKS * (THREADS / 32) * ITERATIONS;
    
    cudaEventRecord(start);
    warp_spinlock_increment<<<BLOCKS, THREADS>>>(d_mutex, d_counter, ITERATIONS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    int result;
    cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("%-25s Count: %-8d Time: %.3f ms\n", "Warp spinlock:", result, ms);
    
    // Compare with pure atomics (all threads)
    cudaMemcpy(d_counter, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    cudaEventRecord(start);
    atomic_increment<<<BLOCKS, THREADS>>>(d_counter, ITERATIONS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("%-25s Count: %-8d Time: %.3f ms\n", "Atomic (all threads):", result, ms);
    
    // Account struct example
    Account* d_account;
    cudaMalloc(&d_account, sizeof(Account));
    Account init_account = {1000, 0};
    cudaMemcpy(d_account, &init_account, sizeof(Account), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mutex, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    printf("\n=== Critical Section Example ===\n");
    printf("Initial account: balance=1000, transactions=0\n");
    
    spinlock_transfer<<<BLOCKS, THREADS>>>(d_mutex, d_account, 5, ITERATIONS);
    cudaDeviceSynchronize();
    
    Account result_account;
    cudaMemcpy(&result_account, d_account, sizeof(Account), cudaMemcpyDeviceToHost);
    
    int expected_transactions = BLOCKS * (THREADS / 32) * ITERATIONS;
    printf("After transfers: balance=%d, transactions=%d\n", 
           result_account.balance, result_account.transactions);
    printf("Expected: balance=%d, transactions=%d\n",
           1000 + expected_transactions * 5, expected_transactions);
    
    // Show the danger
    demonstrate_deadlock_danger<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    printf("\n=== Performance Comparison ===\n");
    printf("Spinlock overhead comes from:\n");
    printf("1. Contention - threads spinning waste cycles\n");
    printf("2. Serialization - only one thread in critical section\n");
    printf("3. Memory traffic - spinning reads memory repeatedly\n\n");
    
    printf("When to use spinlocks on GPU:\n");
    printf("✓ Rarely! Only when atomics can't express your operation\n");
    printf("✓ When critical section has multiple dependent updates\n");
    printf("✓ Always use warp-safe patterns (one locker per warp)\n\n");
    
    printf("Better alternatives:\n");
    printf("• Use atomics (lock-free by design)\n");
    printf("• Pack multiple values into one atomic (64-bit CAS)\n");
    printf("• Redesign algorithm to avoid shared mutable state\n");
    printf("• Use reduction patterns instead of increments\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_mutex);
    cudaFree(d_counter);
    cudaFree(d_account);
    
    return 0;
}
