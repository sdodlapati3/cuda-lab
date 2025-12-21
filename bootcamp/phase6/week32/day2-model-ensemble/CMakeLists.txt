cmake_minimum_required(VERSION 3.18)
project(week32_day2 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(model_ensemble model_ensemble.cu)
