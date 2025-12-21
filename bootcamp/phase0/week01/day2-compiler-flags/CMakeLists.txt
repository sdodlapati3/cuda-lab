cmake_minimum_required(VERSION 3.18)
project(CompilerFlags LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Allow architecture override from command line
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# Build type specific flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Debug: no optimization, device debug symbols
    # WARNING: -G makes kernels very slow!
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -G -g")
    message(STATUS "Debug build: -O0 -G (device debugging, SLOW)")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    # Profile-friendly: optimized but with source mapping
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
    message(STATUS "RelWithDebInfo build: -O3 -lineinfo (for profiling)")
else()
    # Release: maximum optimization
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    message(STATUS "Release build: -O3")
endif()

# Verbose PTX output (see register/shared memory usage)
option(VERBOSE_PTX "Show PTX compilation details" OFF)
if(VERBOSE_PTX)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
endif()

# Create executables for each optimization level experiment
add_executable(vector_add src/vector_add.cu)
