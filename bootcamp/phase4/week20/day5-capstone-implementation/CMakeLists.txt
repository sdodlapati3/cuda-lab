cmake_minimum_required(VERSION 3.18)
project(capstone CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

# Add your capstone executable
add_executable(capstone main.cu)
target_compile_options(capstone PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>)

# Link any required libraries (uncomment as needed)
# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(capstone CUDA::cublas)
