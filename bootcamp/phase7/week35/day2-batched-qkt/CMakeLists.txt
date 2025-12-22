cmake_minimum_required(VERSION 3.18)
project(week35_day2 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(batched_qkt batched_qkt.cu)
