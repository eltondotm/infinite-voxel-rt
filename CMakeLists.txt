cmake_minimum_required(VERSION 3.18)

project(ivr)

find_package(CUDAToolkit)

# select one of these for Debug vs. Release
#set(NVCC_DBG "-g -G")
set (NVCC_DBG "")

set(NVCC_FLAGS "${NVCC_DBG} -m64")
set(GENCODE_FLAGS "-gencode arch=compute_86,code=sm_86")

file(GLOB SRC_FILES "src/main.cu")

add_executable(ivr ${SRC_FILES})

target_compile_options(ivr PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}${GENCODE_FLAGS}")

set_target_properties(ivr PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(ivr PRIVATE CUDA::cudart) 