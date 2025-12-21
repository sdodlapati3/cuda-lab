cmake_minimum_required(VERSION 3.18)
project(bank_conflicts CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)

add_executable(bank_conflicts bank_conflicts.cu)
target_link_libraries(bank_conflicts CUDA::cublas)
target_compile_options(bank_conflicts PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ptxas-options=-v>)
