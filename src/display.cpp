// manifold-rt.cpp : Defines the entry point for the application.
//

#include "display.h"

#include "util/helper_cuda.h"

// Defined in kernel.cu
void createVolume();
void deleteVolume();
void createScene(float3 boundScale);
void deleteScene();
void createBuffers(int width, int height);
void deleteBuffers();
void copyInvViewMatrix(float *matrix, size_t matrixSize);
void renderKernel(uint* d_output, uint bufferW, uint bufferH, float maxDist);

void cleanup()
{
    deletePBO();
    deleteVolume();
    deleteScene();
    deleteBuffers();
}

GLFWwindow* initGL()
{
    GLFWwindow* window;

    if (!glfwInit())
        return nullptr;

    window = glfwCreateWindow(bufferW, bufferH, "Manifold RT", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);

    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
        return nullptr;
    }
    std::cout << glGetString(GL_VERSION) << std::endl;

    glfwSwapInterval(100);

    glfwSetCursorPosCallback(window, cursor);
    glfwSetFramebufferSizeCallback(window, resize);

    return window;
}

void createPBO(uint pbo_res_flags)
{
    if (pbo)
        deletePBO();

    // Create buffer object
    uint size = bufferW * bufferH * 4 * sizeof(GLubyte);
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, size, 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // Register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, pbo_res_flags));

    // Create texture for display
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, bufferW, bufferH, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void deletePBO()
{
    // Unregister this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

    glBindBuffer(1, pbo);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);

    pbo = 0;
    tex = 0;
}

double ox, oy;

void cursor(GLFWwindow* window, double xpos, double ypos)
{
    float dx, dy;
    dx = (float)(xpos - ox);
    dy = (float)(ypos - oy);

    viewRotation.x += dy / 8.0f;
    viewRotation.y += dx / 2.0f;

    ox = xpos;
    oy = ypos;
}


void resize(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
    bufferW = width;
    bufferH = height;

    createPBO(cudaGraphicsMapFlagsWriteDiscard);
    createBuffers(bufferW, bufferH);
}

void render()
{
    copyInvViewMatrix(invViewMatrix, sizeof(float4) * 3);

    uint* d_output;

    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_output, &num_bytes, cuda_pbo_resource));
    //std::cout << "CUDA mapped PBO: May access " << num_bytes << " bytes" << std::endl;

    renderKernel(d_output, bufferW, bufferH, renderDist);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

int main(void)
{
    GLFWwindow* window = initGL();
    if (!window)
        return -1;

    createPBO(cudaGraphicsMapFlagsWriteDiscard);
    createVolume();
    createScene(make_float3(1.2f, 1.0f, 1.2f));
    createBuffers(bufferW, bufferH);

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        GLfloat modelView[16];
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        glTranslatef(viewTranslation.x, viewTranslation.y, viewTranslation.z);
        glRotatef(-viewRotation.y, 0.0f, 1.0f, 0.0f);
        glRotatef(-viewRotation.x, 1.0f, 0.0f, 0.0f);
        glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
        glPopMatrix();

        invViewMatrix[0]  = modelView[0];
        invViewMatrix[1]  = modelView[4];
        invViewMatrix[2]  = modelView[8];
        invViewMatrix[3]  = modelView[12];
        invViewMatrix[4]  = modelView[1];
        invViewMatrix[5]  = modelView[5];
        invViewMatrix[6]  = modelView[9];
        invViewMatrix[7]  = modelView[13];
        invViewMatrix[8]  = modelView[2];
        invViewMatrix[9]  = modelView[6];
        invViewMatrix[10] = modelView[10];
        invViewMatrix[11] = modelView[14];

        render();

        glClear(GL_COLOR_BUFFER_BIT);

        glDisable(GL_DEPTH_TEST);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        // Copy from pbo to texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, bufferW, bufferH, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

        // Draw textured quad
        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0);
        glVertex2f(-1, -1);
        glTexCoord2f(1, 0);
        glVertex2f(1, -1);
        glTexCoord2f(1, 1);
        glVertex2f(1, 1);
        glTexCoord2f(0, 1);
        glVertex2f(-1, 1);
        glEnd();

        glDisable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    cleanup();

    glfwTerminate();
    return 0;
}
