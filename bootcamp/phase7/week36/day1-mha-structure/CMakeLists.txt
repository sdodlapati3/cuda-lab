cmake_minimum_required(VERSION 3.18)
project(week36_day1 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(mha_structure mha_structure.cu)
