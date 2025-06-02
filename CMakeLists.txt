cmake_minimum_required(VERSION 3.18)
project(ivr LANGUAGES CXX CUDA)

find_package(CUDAToolkit)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)  # Replace with your architecture if necessary
endif()

add_executable(ivr src/main.cu)

target_include_directories(ivr PRIVATE CUDA::cudart) 

# From Robert Maynard, "Building Cross-Platform CUDA Applications with CMake"
if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET particle_test 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()