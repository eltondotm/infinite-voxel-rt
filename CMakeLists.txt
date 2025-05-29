cmake_minimum_required(VERSION 3.18)

project(cpt)

find_package(CUDAToolkit)

# select one of these for Debug vs. Release
#set(NVCC_DBG "-g -G")
set (NVCC_DBG "")

set(NVCC_FLAGS "${NVCC_DBG} -m64")
set(GENCODE_FLAGS "-gencode arch=compute_86,code=sm_86")

set(SOURCES main.cu)
set(INCLUDES "")

add_executable(cpt ${SOURCES})

target_compile_options(cpt PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}${GENCODE_FLAGS}")

set_target_properties(cpt PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(cpt PRIVATE CUDA::cudart) 