cmake_minimum_required(VERSION 3.18)
project(MyProject VERSION 1.0.0 LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -lineinfo")
    add_compile_definitions(DEBUG=1)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
endif()

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_EXAMPLES "Build examples" ON)

#-------------------------------------------------------------------------------
# Library
#-------------------------------------------------------------------------------
add_library(myproject STATIC
    src/myproject.cu
    src/cuda_utils.cu
)

target_include_directories(myproject PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

set_target_properties(myproject PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

#-------------------------------------------------------------------------------
# Application
#-------------------------------------------------------------------------------
add_executable(myapp app/main.cpp)
target_link_libraries(myapp PRIVATE myproject)

#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------
if(BUILD_TESTS)
    enable_testing()
    
    add_executable(tests tests/test_main.cpp)
    target_link_libraries(tests PRIVATE myproject)
    target_include_directories(tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    
    add_test(NAME MyProjectTests COMMAND tests)
endif()

#-------------------------------------------------------------------------------
# Benchmarks
#-------------------------------------------------------------------------------
if(BUILD_BENCHMARKS)
    add_executable(benchmarks benchmarks/bench_main.cpp)
    target_link_libraries(benchmarks PRIVATE myproject)
endif()

#-------------------------------------------------------------------------------
# Examples
#-------------------------------------------------------------------------------
if(BUILD_EXAMPLES)
    add_executable(simple_example examples/simple_example.cpp)
    target_link_libraries(simple_example PRIVATE myproject)
endif()

#-------------------------------------------------------------------------------
# Installation
#-------------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS myproject myapp
    EXPORT myproject-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT myproject-targets
    FILE myproject-config.cmake
    NAMESPACE myproject::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/myproject
)
