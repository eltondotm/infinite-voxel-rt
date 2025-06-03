#include "util/error.h"
#include "util/random.h"
#include <iostream>
#include <string>
#include <time.h>

#include "util/vox_loader.h"
#include "sphere.h"
#include "volume.h"
#include "scene.h"
#include "renderer.h"
#include "postprocess.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../deps/stbi/stb_image_write.h"

#define BLOCKSIZE_X 8
#define BLOCKSIZE_Y 8

float render_dist = 500.0f;

const char *volumeFilename = "../../volume.raw";
const char *voxFilename    = "../../mani.vox";
cudaExtent  volumeSize;
typedef unsigned char VolumeType;

cudaArray *d_volumeArray = 0;
cudaTextureObject_t texObject;

// Loading raw texture bytes, use vox_loader for vox files
__host__ void *loadFile(const char *filename, size_t size) {
    FILE *fp = fopen(filename, "rb");

    if (!fp) {
        fprintf(stderr, "Error opening file '%s', errno=%d\n", filename, errno);
        return 0;
    }

    void *data = malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

    return data;
}

__host__ void init_volume(void *h_volume, cudaExtent volume_size) {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
    checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volume_size));

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr =
        make_cudaPitchedPtr(h_volume, volume_size.width * sizeof(VolumeType), volume_size.width, volume_size.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volume_size;
    copyParams.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType         = cudaResourceTypeArray;
    texRes.res.array.array = d_volumeArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;                 // access with normalized texture coordinates
    texDescr.filterMode       = cudaFilterModePoint; // linear interpolation

    texDescr.addressMode[0] = cudaAddressModeClamp;   // clamp texture coordinates
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;

    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL));
}

__global__ void print_volume(cudaTextureObject_t tex, cudaExtent dims) {
    for (int z = 0; z < dims.depth; ++z) {
        for (int y = 0; y < dims.height; ++y) {
            for (int x = 0; x < dims.width; ++x) {
                VolumeType val = tex3D<VolumeType>(tex, (float)x, (float)y, (float)z);
                if (val != 0) {
                    printf("(%i,%i,%i)->%i", x, y, z, val);
                }
            }
        }
    }
}

__global__ void render(Renderer *ren, float render_dist) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int w = ren->out.w;
    int h = ren->out.h;
    if((i >= w) || (j >= h)) return;
    int pixel_index = (h-j-1)*w + i;
    float u = float(i) / float(w);
    float v = float(j) / float(h);
    Ray r = ren->cam.generate_ray(u, v);
    HitRecord buff = ren->trace_ray(r, render_dist);
    ren->gb[pixel_index] = make_float4(buff.t, buff.normal.x, buff.normal.y, buff.normal.z);
    ren->out.fb[pixel_index] = ren->buff_to_color(r, buff, render_dist);
}

__global__ void init_renderer(Renderer *renderer, Vec3 *fb, float4 *gb, int w, int h, 
                              Hitable **scene, BBox **world_bounds, curandState *rand_state) {
    Image out(fb, w, h);
    *renderer = Renderer(out, gb, scene, world_bounds, rand_state);
}

__global__ void create_scene(Hitable **d_list, 
                             Hitable **d_world, 
                             BBox **d_world_bounds, 
                             cudaTextureObject_t tex,
                             cudaExtent size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        //*(d_list)   = new Sphere(Vec3(-1), 0.5);
        *(d_list)   = new Volume(tex, size);
        *d_world    = new Scene(d_list, 1);

        Vec3 min_extent(0);
        Vec3 max_extent(size.width*1.1f, size.height, size.depth*1.1f);
        *d_world_bounds = new BBox(min_extent, max_extent);
    }
}

__global__ void free_scene(Hitable **d_list, Hitable **d_world, BBox **d_world_bounds) {
    delete *(d_list);
    //delete *(d_list+1);
    delete *d_world;
    delete *d_world_bounds;
}

int main(int argc, char* argv[]) {
    int nx = 3840, ny = 2160;
    if (argc == 3) {
        nx = std::stoi(argv[1]);
        ny = std::stoi(argv[2]);
    }
    if (argc == 2) {
        render_dist = std::stof(argv[1]);
    }

    std::cout << "Rendering a " << nx << "x" << ny << " image ";
    std::cout << "in " << BLOCKSIZE_X << "x" << BLOCKSIZE_Y << " blocks.\n";

    int num_pixels = nx*ny;
    int num_pixels_out = num_pixels/4;
    size_t fb_size = num_pixels*sizeof(Vec3);
    size_t gb_size = num_pixels*sizeof(float4);
    size_t out_size = num_pixels_out*3*sizeof(uint8_t);

    // If loading raw files, make sure to initialize volumeSize ahead of time
    // size_t size = volumeSize.width * volumeSize.height * volumeSize.depth * sizeof(VolumeType);
    // void *h_volume = loadFile(volumeFilename, size);

    void *h_volume = loadVox(voxFilename, volumeSize);
    init_volume(h_volume, volumeSize);
    //print_volume<<<1,1>>>(texObject, volumeSize);
    //SYNC

    Hitable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 2*sizeof(Hitable *)));
    Hitable **d_scene;
    checkCudaErrors(cudaMalloc((void **)&d_scene, sizeof(Hitable *)));
    BBox **d_world_bounds;
    checkCudaErrors(cudaMallocManaged((void**)&d_world_bounds, sizeof(BBox)));
    create_scene<<<1,1>>>(d_list, d_scene, d_world_bounds, texObject, volumeSize);
    SYNC

    Vec3 *fb;
    float4 *gb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
    checkCudaErrors(cudaMallocManaged((void **)&gb, gb_size));

    dim3 blocks(nx / BLOCKSIZE_X + 1, ny / BLOCKSIZE_Y + 1);
    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y);

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    init_rng<<<blocks, threads>>>(nx, ny, d_rand_state);
    SYNC
    Renderer *ren;
    checkCudaErrors(cudaMallocManaged((void **)&ren, sizeof(Renderer)));
    init_renderer<<<1,1>>>(ren, fb, gb, nx, ny, d_scene, d_world_bounds, d_rand_state);
    SYNC

    clock_t start, stop;
    start = clock();
    render<<<blocks,threads>>>(ren, render_dist);
    SYNC
    edge_detect<<<blocks,threads>>>(fb, gb, nx, ny);
    SYNC
    fog<<<blocks,threads>>>(fb, gb, render_dist, nx, ny);
    SYNC
    vignette<<<blocks,threads>>>(fb, nx, ny);
    SYNC
    stop = clock();
    double timer_s = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cout << "took " << timer_s << " seconds.\n";

    // Output image setup and cleanup
    dim3 blocks_out(nx/2 / BLOCKSIZE_X + 1, ny/2 / BLOCKSIZE_Y + 1);
    uint8_t *out;
    checkCudaErrors(cudaMallocManaged((void **)&out, out_size));
    downsample<<<blocks_out,threads>>>(out, fb, nx, ny);
    SYNC

    int n_channels = 3;
    stbi_write_png("out.png", nx/2, ny/2, n_channels, out, nx/2*n_channels*sizeof(uint8_t));

    free_scene<<<1, 1>>>(d_list, d_scene, d_world_bounds);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(gb));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_world_bounds));
    checkCudaErrors(cudaFree(out));
    checkCudaErrors(cudaFree(ren));

    cudaDeviceReset();
    return 0;
}