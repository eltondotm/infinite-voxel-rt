# CMakeList.txt : CMake project for manifold-rt, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.10)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("manifold-rt" LANGUAGES CXX CUDA)

find_package(OpenGL REQUIRED) 
if (CMAKE_VERSION VERSION_LESS 3.17)
	find_package(CUDA REQUIRED)
else()
	find_package(CUDAToolkit REQUIRED)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

set(SRCS 
	"src/display.cpp" 
	"src/kernel.cu"
)

#set(DEPS
#   "deps/GLFW/glfw3.h" 
#	"deps/GLFW/glfw3native.h" 
#	"deps/GLEW/eglew.h" 
#	"deps/GLEW/glew.h" 
#	"deps/GLEW/glxew.h" 
#	"deps/GLEW/wglew.h" 
#)

# Add source to this project's executable.
add_executable (manifold-rt ${SRCS} ${DEPS})

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET manifold-rt PROPERTY CXX_STANDARD 20)
endif()

target_compile_options(manifold-rt PRIVATE --extended-lambda)

add_compile_definitions(GLEW_STATIC)

set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/lib/)

target_include_directories(manifold-rt PRIVATE deps ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(manifold-rt PRIVATE OpenGL::GL ${LIB_DIR}glfw3.lib ${LIB_DIR}glew32s.lib)
