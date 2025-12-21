cmake_minimum_required(VERSION 3.18)
project(week27_day4 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(warp_spec_tc warp_spec_tc.cu)
