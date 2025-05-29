#pragma once

#include <iostream>
#include "vec3.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

#define SYNC checkCudaErrors(cudaGetLastError()); checkCudaErrors(cudaDeviceSynchronize());

__global__ void ldr_to_int(uint8_t* out, Vec3 *in, size_t size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= size) return;
    out[idx*3]   = in[idx][0] * 255.99;
    out[idx*3+1] = in[idx][1] * 255.99;
    out[idx*3+2] = in[idx][2] * 255.99;
}
