Title: 5.1. Compute Capabilities — CUDA Programming Guide

URL Source: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html

Published Time: Mon, 15 Dec 2025 20:02:32 GMT

Markdown Content:
5.1. Compute Capabilities[#](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities "Link to this heading")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------

The general specifications and features of a compute device depend on its compute capability (see [Compute Capability and Streaming Multiprocessor Versions](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html.md#cuda-platform-compute-capability-sm-version)).

[Table 29](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-table-features-and-technical-specifications-feature-support-per-compute-capability), [Table 30](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-table-device-and-streaming-multiprocessor-sm-information-per-compute-capability), and [Table 31](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-table-memory-information-per-compute-capability) show the features and technical specifications associated with each compute capability that is currently supported.

All NVIDIA GPU architectures use a little-endian representation.

5.1.1. Obtain the GPU Compute Capability[#](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#obtain-the-gpu-compute-capability "Link to this heading")
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) page provides a comprehensive mapping from NVIDIA GPU models to their compute capability.

Alternatively, the [nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html.md) tool, provided with the [NVIDIA Driver](https://www.nvidia.com/en-us/drivers.md/), can be used to get the compute capability of a GPU. For example, the following command will output the GPU names and compute capabilities available on the system:

nvidia-smi --query-gpu=name,compute_cap

At runtime, the compute capability can be obtained using the CUDA Runtime API [cudaDeviceGetAttribute()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html.md#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151) , CUDA Driver API [cuDeviceGetAttribute()](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html.md#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266), or NVML API [nvmlDeviceGetCudaComputeCapability()](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html.md#group__nvmlDeviceQueries_1g1f803a2fb4b7dfc0a8183b46b46ab03a):

#include <cuda_runtime_api.h>

int computeCapabilityMajor, computeCapabilityMinor;
cudaDeviceGetAttribute(&computeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, device_id);
cudaDeviceGetAttribute(&computeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, device_id);

#include <cuda.h>

int computeCapabilityMajor, computeCapabilityMinor;
cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_id);
cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_id);

#include <nvml.h> // required linking with -lnvidia-ml

int computeCapabilityMajor, computeCapabilityMinor;
nvmlDeviceGetCudaComputeCapability(nvmlDevice, &computeCapabilityMajor, &computeCapabilityMinor);

5.1.2. Feature Availability[#](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#feature-availability "Link to this heading")
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Most compute features introduced with a compute architecture are intended to be available on all subsequent architectures. This is shown in [Table 29](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-table-features-and-technical-specifications-feature-support-per-compute-capability) by the “yes” for availability of a feature on compute capabilities subsequent to its introduction.

### 5.1.2.1. Architecture-Specific Features[#](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#architecture-specific-features "Link to this heading")

Beginning with devices of Compute Capability 9.0, specialized compute features that are introduced with an architecture may not be guaranteed to be available on all subsequent compute capabilities. These features are called _architecture-specific_ features and target acceleration of specialized operations, such as Tensor Core operations, which are not intended for all classes of compute capabilities or may significantly change in future generations. Code must be compiled with an architecture-specific compiler target (see [Feature Set Compiler Targets](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-feature-set-compiler-targets)) to enable architecture-specific features. Code compiled with an architecture-specific compiler target can only be run on the exact compute capability it was compiled for.

### 5.1.2.2. Family-Specific Features[#](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#family-specific-features "Link to this heading")

Beginning with devices of Compute Capability 10.0, some architecture-specific features are common to devices of more than one compute capability. The devices that contain these features are part of the same family and these features can also be called _family-specific_ features. Family-specific features are guaranteed to be available on all devices in the same family. A family-specific compiler target is required to enable family-specific features. See [Section 5.1.2.3](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-feature-set-compiler-targets). Code compiled for a family-specific target can only be run on GPUs which are members of that family.

### 5.1.2.3. Feature Set Compiler Targets[#](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#feature-set-compiler-targets "Link to this heading")

There are three sets of compute features which the compiler can target:

**Baseline Feature Set**: The predominant set of compute features that are introduced with the intent to be available for subsequent compute architectures. These features and their availability are summarized in [Table 29](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-table-features-and-technical-specifications-feature-support-per-compute-capability).

**Architecture-Specific Feature Set**: A small and highly specialized set of features called architecture-specific, that are introduced to accelerate specialized operations, which are not guaranteed to be available or might change significantly on subsequent compute architectures. These features are summarized in the respective “Compute Capability #.#” subsections. The architecture-specific feature set is a superset of the family-specific feature set. Architecture-specific compiler targets were introduced with Compute Capability 9.0 devices and are selected by using an **a** suffix in the compilation target, for example by specifying `compute_100a` or `compute_120a` as the compute target.

**Family-Specific Feature Set**: Some architecture-specific features are common to GPUs of more than one compute capability. These features are summarized in the respective “Compute Capability #.#” subsections. With a few exceptions, later-generation devices with the same major compute capability are in the same family. [Table 28](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-family-specific-compatibility) indicates the compatibility of family-specific targets with device compute capability, including exceptions. The family-specific feature set is a superset of the baseline feature set. Family-specific compiler targets were introduced with Compute Capability 10.0 devices and are selected by using an **f** suffix in the compilation target, for example by specifying `compute_100f` or `compute_120f` as the compute target.

All devices starting from compute capability 9.0 have a set of features that are architecture-specific. To utilize the complete set of these features on a specific GPU, the architecture-specific compiler target with the suffix **a** must be used. Additionally, starting from compute capability 10.0, there are sets of features that appear in multiple devices with different minor compute capabilities. These sets of instructions are called family-specific features, and the devices which share these features are said to be part of the same family. The family-specific features are a subset of the architecture-specific features that are shared by all members of that GPU family. The family-specific compiler target with the suffix **f** allows the compiler to generate code that uses this common subset of architecture-specific features.

For example:

*   The `compute_100` compilation target does not allow the use of architecture-specific features. This target will be compatible with all devices of compute capability 10.0 and later.

*   The `compute_100f`_family-specific_ compilation target allows the use of the subset of architecture-specific features that are common across the GPU family. This target will only be compatible with devices that are part of the GPU family. In this example, it is compatible with devices of Compute Capability 10.0 and Compute Capability 10.3. The features available in the family-specific `compute_100f` target are a superset of the features available in the baseline `compute_100` target.

*   The `compute_100a`_architecture-specific_ compilation target allows the use of the complete set of architecture-specific features in Compute Capability 10.0 devices. This target will only be compatible with devices of Compute Capability 10.0 and no others. The features available in the `compute_100a` target form a superset of the features available in the `compute_100f` target.

Table 28 Family-Specific Compatibility[#](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-family-specific-compatibility "Link to this table")| Compilation Target | Compatible with Compute Capability |
| --- | --- |
| `compute_100f` | 10.0 | 10.3 |
| `compute_103f` | 10.3 [[1]](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#family2) |
| `compute_110f` | 11.0 [[1]](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#family2) |
| `compute_120f` | 12.0 | 12.1 |
| `compute_121f` | 12.1 [[1]](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#family2) |

5.1.3. Features and Technical Specifications[#](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#features-and-technical-specifications "Link to this heading")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Table 29 Feature Support per Compute Capability[#](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-table-features-and-technical-specifications-feature-support-per-compute-capability "Link to this table")| **Feature Support** | **Compute Capability** |
| --- | --- |
| (Unlisted features are supported for all compute capabilities) | 7.x | 8.x | 9.0 | 10.x | 11.0 | 12.x |
| Atomic functions operating on 128-bit integer values in shared and global memory ([Atomic Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.md#atomic-functions)) | No | Yes |
| Atomic addition operating on `float2` and `float4` floating point vectors in global memory ([atomicAdd()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.md#atomicadd)) | No | Yes |
| Warp reduce functions ([Warp Reduce Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.md#warp-reduce-functions)) | No | Yes |
| Bfloat16-precision floating-point operations | No | Yes |
| 128-bit-precision floating-point operations | No | Yes |
| Hardware-accelerated `memcpy_async` ([Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html.md#pipelines)) | No | Yes |
| Hardware-accelerated Split Arrive/Wait Barrier ([Asynchronous Barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html.md#asynchronous-barriers)) | No | Yes |
| L2 Cache Residency Management ([L2 Cache Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html.md#advanced-kernels-l2-control)) | No | Yes |
| DPX Instructions for Accelerated Dynamic Programming ([Dynamic Programming eXtension (DPX) Instructions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.md#dpx-instructions)) | Multiple Instr. | Native | Multiple Instr. |
| Distributed Shared Memory | No | Yes |
| Thread Block Cluster ([Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html.md#thread-block-clusters)) | No | Yes |
| Tensor Memory Accelerator (TMA) unit ([Using the Tensor Memory Accelerator (TMA)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html.md#async-copies-tma)) | No | Yes |

Note that the KB and K units used in the following tables correspond to 1024 bytes (i.e., a KiB) and 1024 respectively.

Table 30 Device and Streaming Multiprocessor (SM) Information per Compute Capability[#](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-table-device-and-streaming-multiprocessor-sm-information-per-compute-capability "Link to this table")|  | **Compute Capability** |
| --- | --- |
|  | 7.5 | 8.0 | 8.6 | 8.7 | 8.9 | 9.0 | 10.0 | 10.3 | 11.0 | 12.x |
| Ratio of FP32 to FP64 Throughput [[2]](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#fn-cc-throughput) | 32:1 | 2:1 | 64:1 | 2:1 | 64:1 |
| Maximum number of resident grids per device (Concurrent Kernel Execution) | 128 |
| Maximum dimensionality of a grid | 3 |
| Maximum x-dimension of a grid | 2 31-1 |
| Maximum y- or z-dimension of a grid | 65535 |
| Maximum dimensionality of a thread block | 3 |
| Maximum x- or y-dimensionality of a thread block | 1024 |
| Maximum z-dimension of a thread block | 64 |
| Maximum number of threads per block | 1024 |
| Warp size | 32 |
| Maximum number of resident blocks per SM | 16 | 32 | 16 | 24 | 32 | 24 |
| Maximum number of resident warps per SM | 32 | 64 | 48 | 64 | 48 |
| Maximum number of resident threads per SM | 1024 | 2048 | 1536 | 2048 | 1536 |
| Green contexts: minimum SM partition size for useFlags 0 | 2 | 4 | 8 |
| Green contexts: SM co-scheduled alignment per partition for useFlags 0 | 2 | 8 |

Table 31 Memory Information per Compute Capability[#](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-table-memory-information-per-compute-capability "Link to this table")|  | **Compute Capability** |
| --- | --- |
|  | 7.5 | 8.0 | 8.6 | 8.7 | 8.9 | 9.0 | 10.x | 11.0 | 12.x |
| Number of 32-bit registers per SM | 64 K |
| Maximum number of 32-bit registers per thread block | 64 K |
| Maximum number of 32-bit registers per thread | 255 |
| Maximum amount of shared memory per SM | 64 KB | 164 KB | 100 KB | 164 KB | 100 KB | 228 KB | 100 KB |
| Maximum amount of shared memory per thread block [[3]](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#fn33) | 64 KB | 163 KB | 99 KB | 163 KB | 99 KB | 227 KB | 99 KB |
| Number of shared memory banks | 32 |
| Maximum amount of local memory per thread | 512 KB |
| Constant memory size | 64 KB |
| Cache working set per SM for constant memory | 8 KB |
| Cache working set per SM for texture memory | 32 or 64 KB | 28 KB ~ 192 KB | 28 KB ~ 128 KB | 28 KB ~ 192 KB | 28 KB ~ 128 KB | 28 KB ~ 256 KB | 28 KB ~ 128 KB |

Table 32 Shared Memory Capacity per Compute Capability[#](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-table-shared-memory-capacity-per-compute-capability "Link to this table")| Compute Capability | Unified Data Cache Size (KB) | SMEM Capacity Sizes (KB) |
| --- | --- | --- |
| 7.5 | 96 | 32, 64 |
| 8.0 | 192 | 0, 8, 16, 32, 64, 100, 132, 164 |
| 8.6 | 128 | 0, 8, 16, 32, 64, 100 |
| 8.7 | 192 | 0, 8, 16, 32, 64, 100, 132, 164 |
| 8.9 | 128 | 0, 8, 16, 32, 64, 100 |
| 9.0 | 256 | 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 |
| 10.x | 256 | 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 |
| 11.0 | 256 | 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 |
| 12.x | 128 | 0, 8, 16, 32, 64, 100 |

[Table 33](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-table-tensor-core-data-types-per-compute-capability) shows the input data types supported by Tensor Core acceleration. The Tensor Core feature set is available within the CUDA compilation toolchain through inline PTX. It is strongly recommended that applications use this feature set through CUDA-X libraries such as cuDNN, cuBLAS, and cuFFT, for example, or through [CUTLASS](https://docs.nvidia.com/cutlass/index.html), a collection of CUDA C++ template abstractions and Python domain-specific languages (DSLs) designed to enable high-performance matrix-matrix multiplication (GEMM) and related computations across all levels within CUDA.

Table 33 Input Data Types Supported by Tensor Core Acceleration per Compute Capability[#](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-table-tensor-core-data-types-per-compute-capability "Link to this table")| Compute Capability | Tensor Core Input Data Types |
| --- | --- |
|  | FP64 | TF32 | BF16 | FP16 | FP8 | FP6 | FP4 | INT8 | INT4 |
| 7.5 |  | Yes |  | Yes | Yes |
| 8.0 | Yes | Yes | Yes | Yes |  | Yes | Yes |
| 8.6 |  | Yes | Yes | Yes |  | Yes | Yes |
| 8.7 |  | Yes | Yes | Yes |  | Yes | Yes |
| 8.9 |  | Yes | Yes | Yes | Yes |  | Yes | Yes |
| 9.0 | Yes | Yes | Yes | Yes | Yes |  | Yes |  |
| 10.0 | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |  |
| 10.3 |  | Yes | Yes | Yes | Yes | Yes | Yes | Yes |  |
| 11.0 |  | Yes | Yes | Yes | Yes | Yes | Yes | Yes |  |
| 12.x |  | Yes | Yes | Yes | Yes | Yes | Yes | Yes |  |

Links/Buttons:
- [1](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#id2)
- [2](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#id5)
- [3](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#id6)
- [#](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-table-shared-memory-capacity-per-compute-capability)
- [Compute Capability and Streaming Multiprocessor Versions](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html.md#cuda-platform-compute-capability-sm-version)
- [Table 29](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-table-features-and-technical-specifications-feature-support-per-compute-capability)
- [Table 30](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-table-device-and-streaming-multiprocessor-sm-information-per-compute-capability)
- [Table 31](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-table-memory-information-per-compute-capability)
- [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)
- [nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html.md)
- [NVIDIA Driver](https://www.nvidia.com/en-us/drivers.md/)
- [cudaDeviceGetAttribute()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html.md#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151)
- [cuDeviceGetAttribute()](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html.md#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266)
- [nvmlDeviceGetCudaComputeCapability()](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html.md#group__nvmlDeviceQueries_1g1f803a2fb4b7dfc0a8183b46b46ab03a)
- [Feature Set Compiler Targets](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-feature-set-compiler-targets)
- [Table 28](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-family-specific-compatibility)
- [[1]](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#family2)
- [Atomic Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.md#atomic-functions)
- [atomicAdd()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.md#atomicadd)
- [Warp Reduce Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.md#warp-reduce-functions)
- [Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html.md#pipelines)
- [Asynchronous Barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html.md#asynchronous-barriers)
- [L2 Cache Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html.md#advanced-kernels-l2-control)
- [Dynamic Programming eXtension (DPX) Instructions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.md#dpx-instructions)
- [Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html.md#thread-block-clusters)
- [Using the Tensor Memory Accelerator (TMA)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html.md#async-copies-tma)
- [[2]](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#fn-cc-throughput)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html.md#arithmetic-instructions-throughput-native-arithmetic-instructions)
- [[3]](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#fn33)
- [Configuring L1/Shared Memory Balance](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html.md#advanced-kernel-l1-shared-config)
- [Table 33](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html.md#compute-capabilities-table-tensor-core-data-types-per-compute-capability)
- [CUTLASS](https://docs.nvidia.com/cutlass/index.html)
