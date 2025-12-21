cmake_minimum_required(VERSION 3.18)
project(week31_day6 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(inference_summary inference_summary.cu)
