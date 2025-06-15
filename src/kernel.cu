#pragma once

#include "util/helper_cuda.h"
#include "util/helper_math.h"
#include "util/vox.h"

#include "renderer.h"
#include "postprocess.h"

cudaArray* d_volumeArray;
cudaTextureObject_t texObject;

const char* volumeFilename = "../../../mani.vox";
cudaExtent  volumeSize;

Scene  *d_scene;
float4 *d_gBuffer = nullptr;

__constant__ float3x4 c_invViewMatrix;

void createVolume() {
	void *h_volume = loadVox(volumeFilename, volumeSize);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();
    checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr =
        make_cudaPitchedPtr(h_volume, volumeSize.width * sizeof(uchar), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType         = cudaResourceTypeArray;
    texRes.res.array.array = d_volumeArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = cudaFilterModePoint;

    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;

    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL));
	free(h_volume);
}

void deleteVolume() {
	checkCudaErrors(cudaDestroyTextureObject(texObject));
	checkCudaErrors(cudaFreeArray(d_volumeArray));
}

__global__ void d_updateScene(Scene* scene, 
							  cudaTextureObject_t tex, 
							  cudaExtent dims, 
							  float3 boundScale) {
	Volume vol;
	vol.tex  = tex;
	vol.dims = dims;
	float3 min = make_float3(0.0f);
	float3 max = make_float3(vol.dims.width, vol.dims.height, vol.dims.depth);
	vol.box  = BBox(min, max);

	scene->volume = vol;
	scene->bounds = BBox(min, max * boundScale);
}

void deleteScene() {
	checkCudaErrors(cudaFree(d_scene));
}

void updateScene(float3 boundScale) {
	d_updateScene<<<1, 1>>>(d_scene, texObject, volumeSize, boundScale);
}

void createScene(float3 boundScale) {
	checkCudaErrors(cudaMallocManaged((void**)&d_scene, sizeof(Scene)));
	updateScene(boundScale);
}

void deleteBuffers() {
	checkCudaErrors(cudaFree(d_gBuffer));
}

void createBuffers(int width, int height) {
	if (d_gBuffer) {
		deleteBuffers();
	}
	size_t size = width * height * sizeof(float4);
	checkCudaErrors(cudaMallocManaged((void**)&d_gBuffer, size));
}

__global__ void d_render(uint*   d_output,
						 float4* d_gBuffer,
						 uint    width,
						 uint    height,
						 float   maxDist,
						 Scene*  scene) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	float u = (float)x / (float)width  * 2.0f - 1.0f;
	float v = (float)y / (float)height * 2.0f - 1.0f;

	float3 eyePos = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
	float3 eyeDir = normalize(mul(c_invViewMatrix, normalize(make_float3(u, v, -2.0f))));
	Ray r(eyePos, eyeDir);
	
	// Get scene intersection information
	HitRecord rec = traceRay(r, scene, maxDist);

	// Determine color based on the surface hit
	float4 color = make_float4(buffToColor(r, rec, scene, maxDist), 1.0f);

	d_output[y * width + x] = rgbaFloatToInt(color);
	d_gBuffer[y * width + x] = make_float4(rec.normal, rec.t);
}

void renderKernel(uint* d_output, uint width, uint height, float maxDist)
{
	uint block_x = 16;
	uint block_y = 16;

	dim3 block(block_x, block_y);
	dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y);

	d_render<<<grid, block>>>(d_output, d_gBuffer, width, height, maxDist, d_scene);
	d_edgeDetect<<<grid, block>>>(d_output, d_gBuffer, width, height);
	d_fog<<<grid, block>>>(d_output, d_gBuffer, maxDist, width, height);
	d_vignette<<<grid, block>>>(d_output, width, height);
}

void copyInvViewMatrix(float *matrix, size_t matrixSize) {
	checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, matrix, matrixSize));
}