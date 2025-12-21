cmake_minimum_required(VERSION 3.18)
project(week30_day5 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(tf_custom_op tf_custom_op.cu)
