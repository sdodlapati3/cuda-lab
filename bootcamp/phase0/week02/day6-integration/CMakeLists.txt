cmake_minimum_required(VERSION 3.18)
project(DebugIntegration LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# Debug vs Release configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug build: enabling -G -g for device debugging")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -lineinfo")
    add_compile_definitions(DEBUG=1)
else()
    message(STATUS "Release build: enabling -O3")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
endif()

add_executable(buggy_app src/buggy_app.cu)
add_executable(debugged_app src/debugged_app.cu)
