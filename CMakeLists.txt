cmake_minimum_required(VERSION 3.18)
project(ivr LANGUAGES CXX CUDA)

find_package(CUDAToolkit)

# select one of these for Debug vs. Release
#set(NVCC_DBG "-g -G")
set (NVCC_DBG "")

set(NVCC_FLAGS "${NVCC_DBG} -m64")
set(GENCODE_FLAGS "-gencode arch=compute_86,code=sm_86")

add_executable(ivr src/main.cu)

target_compile_options(ivr PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}${GENCODE_FLAGS}")

set_property(TARGET ivr
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(ivr PRIVATE CUDA::cudart) 

# From Robert Maynard, "Building Cross-Platform CUDA Applications with CMake"
if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET particle_test 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()