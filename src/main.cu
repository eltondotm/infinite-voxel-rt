#include "util/error.h"
#include "util/random.h"
#include <iostream>
#include <string>
#include <time.h>

#include "sphere.h"
#include "scene.h"
#include "renderer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../deps/stbi/stb_image_write.h"

#define BLOCKSIZE_X 8
#define BLOCKSIZE_Y 8

__global__ void render(Renderer *ren) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int w = ren->out.w;
    int h = ren->out.h;
    if((i >= w) || (j >= h)) return;
    int pixel_index = (h-j-1)*w + i;
    float u = float(i) / float(w);
    float v = float(j) / float(h);
    Ray r = ren->cam.generate_ray(u, v);
    ren->out.fb[pixel_index] = ren->trace_ray(r);
}

__global__ void init_renderer(Renderer *renderer, Vec3 *fb, int w, int h, 
                              Hitable **scene, curandState *rand_state) {
    Image out(fb, w, h);
    *renderer = Renderer(out, scene, rand_state);
}

__global__ void create_scene(Hitable **d_list, Hitable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new Sphere(Vec3(0,0,-1), 0.5);
        *d_world    = new Scene(d_list, 1);
    }
}

__global__ void free_scene(Hitable **d_list, Hitable **d_world) {
    delete *(d_list);
    //delete *(d_list+1);
    delete *d_world;
}

int main(int argc, char* argv[]) {
    int nx = 1200, ny = 600;
    int ns = 100;
    if (argc == 3) {
        nx = std::stoi(argv[1]);
        ny = std::stoi(argv[2]);
    }

    std::cout << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cout << "in " << BLOCKSIZE_X << "x" << BLOCKSIZE_Y << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(Vec3);
    size_t out_size = num_pixels*3*sizeof(uint8_t);

    Hitable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 2*sizeof(Hitable *)));
    Hitable **d_scene;
    checkCudaErrors(cudaMalloc((void **)&d_scene, sizeof(Hitable *)));
    create_scene<<<1,1>>>(d_list, d_scene);
    SYNC

    Vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    dim3 blocks(nx / BLOCKSIZE_X + 1, ny / BLOCKSIZE_Y + 1);
    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y);

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    init_rng<<<blocks, threads>>>(nx, ny, d_rand_state);
    SYNC
    Renderer *ren;
    checkCudaErrors(cudaMallocManaged((void **)&ren, sizeof(Renderer)));
    init_renderer<<<1,1>>>(ren, fb, nx, ny, d_scene, d_rand_state);
    SYNC

    clock_t start, stop;
    start = clock();
    render<<<blocks,threads>>>(ren);
    SYNC
    stop = clock();
    double timer_s = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cout << "took " << timer_s << " seconds.\n";

    // Output image setup and cleanup
    int block_size = 256;
    int num_blocks = (num_pixels*3 + block_size - 1) / block_size;
    
    uint8_t *out;
    checkCudaErrors(cudaMallocManaged((void **)&out, out_size));
    ldr_to_int<<<num_blocks, block_size>>>(out, fb, num_pixels);
    SYNC

    int n_channels = 3;
    stbi_write_png("out.png", nx, ny, n_channels, out, nx*n_channels*sizeof(uint8_t));

    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(out));
    checkCudaErrors(cudaFree(ren));

    cudaDeviceReset();
    return 0;
}