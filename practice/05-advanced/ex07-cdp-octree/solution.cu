#include <stdio.h>
#include <cuda_runtime.h>

struct Point3D {
    float x, y, z;
};

struct BoundingBox {
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
};

__device__ int countPointsInBox(Point3D* points, int n, BoundingBox box) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (points[i].x >= box.minX && points[i].x < box.maxX &&
            points[i].y >= box.minY && points[i].y < box.maxY &&
            points[i].z >= box.minZ && points[i].z < box.maxZ) {
            count++;
        }
    }
    return count;
}

__device__ BoundingBox getChildBox(BoundingBox parent, int octant) {
    float midX = (parent.minX + parent.maxX) / 2;
    float midY = (parent.minY + parent.maxY) / 2;
    float midZ = (parent.minZ + parent.maxZ) / 2;
    
    BoundingBox child;
    child.minX = (octant & 1) ? midX : parent.minX;
    child.maxX = (octant & 1) ? parent.maxX : midX;
    child.minY = (octant & 2) ? midY : parent.minY;
    child.maxY = (octant & 2) ? parent.maxY : midY;
    child.minZ = (octant & 4) ? midZ : parent.minZ;
    child.maxZ = (octant & 4) ? parent.maxZ : midZ;
    
    return child;
}

__global__ void buildOctree(Point3D* points, int n, BoundingBox box, 
                            int* nodeCount, int depth, int maxDepth, int threshold) {
    int count = countPointsInBox(points, n, box);
    
    if (threadIdx.x == 0) {
        atomicAdd(nodeCount, 1);
    }
    
    if (count <= threshold || depth >= maxDepth) {
        return;
    }
    
    if (threadIdx.x == 0) {
        cudaStream_t streams[8];
        for (int i = 0; i < 8; i++) {
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        }
        
        for (int octant = 0; octant < 8; octant++) {
            BoundingBox childBox = getChildBox(box, octant);
            buildOctree<<<1, 1, 0, streams[octant]>>>(points, n, childBox, nodeCount, 
                                                       depth + 1, maxDepth, threshold);
        }
        
        cudaDeviceSynchronize();
        
        for (int i = 0; i < 8; i++) {
            cudaStreamDestroy(streams[i]);
        }
    }
}

int main() {
    const int N = 1000;
    Point3D* h_points = new Point3D[N];
    
    srand(42);
    for (int i = 0; i < N; i++) {
        h_points[i].x = (float)rand() / RAND_MAX;
        h_points[i].y = (float)rand() / RAND_MAX;
        h_points[i].z = (float)rand() / RAND_MAX;
    }
    
    Point3D* d_points;
    int* d_nodeCount;
    cudaMalloc(&d_points, N * sizeof(Point3D));
    cudaMalloc(&d_nodeCount, sizeof(int));
    cudaMemcpy(d_points, h_points, N * sizeof(Point3D), cudaMemcpyHostToDevice);
    cudaMemset(d_nodeCount, 0, sizeof(int));
    
    BoundingBox rootBox = {0, 0, 0, 1, 1, 1};
    int maxDepth = 5;
    int threshold = 10;
    
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, maxDepth + 1);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
    
    buildOctree<<<1, 1>>>(d_points, N, rootBox, d_nodeCount, 0, maxDepth, threshold);
    cudaDeviceSynchronize();
    
    int h_nodeCount;
    cudaMemcpy(&h_nodeCount, d_nodeCount, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Points: %d\n", N);
    printf("Max depth: %d\n", maxDepth);
    printf("Threshold: %d\n", threshold);
    printf("Nodes created: %d\n", h_nodeCount);
    printf("Test %s\n", (h_nodeCount > 1) ? "PASSED" : "FAILED");
    
    delete[] h_points;
    cudaFree(d_points);
    cudaFree(d_nodeCount);
    
    return (h_nodeCount > 1) ? 0 : 1;
}
