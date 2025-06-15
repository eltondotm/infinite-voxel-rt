// manifold-rt.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <iostream>

#include <GLEW/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

typedef unsigned int  uint;
typedef unsigned char uchar;

// Framebuffer size
uint bufferW = 1280;
uint bufferH = 720;

// View matrix
float3 viewRotation;
float3 viewTranslation = make_float3(60.0f, 42.0f, 78.0f);
float  invViewMatrix[12];

// Render parameters
float renderDist = 500.0f;

// Pixel buffer object variables
GLuint                       pbo = 0;
GLuint                       tex = 0;
struct cudaGraphicsResource* cuda_pbo_resource;

void cleanup();

// GL functions
GLFWwindow* initGL();
void createPBO(uint pbo_res_flags);
void deletePBO();

// Rendering callbacks
void cursor(GLFWwindow* window, double xpos, double ypos);
void resize(GLFWwindow* window, int width, int height);

// CUDA functions
void render();