#pragma once

#include "vec3.h"
#include "util/device_util.h"

inline __device__ Vec3 normalize_n(const Vec3& n) {
    return (n * 2.0f - 1.0f).unit();
}

__global__ void edge_detect(Vec3 *fb, float4 *gb, size_t nx, size_t ny) {
    const Vec3 sobel_x0(1.0f, 0.0f, -1.0f);
    const Vec3 sobel_x1(2.0f, 0.0f, -2.0f);
    const Vec3 sobel_x2(1.0f, 0.0f, -1.0f);

    const Vec3 sobel_y0(1.0f,   2.0f,  1.0f);
    const Vec3 sobel_y1(0.0f,   0.0f,  0.0f);
    const Vec3 sobel_y2(-1.0f, -2.0f, -1.0f);

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= nx) || (j >= ny)) return;

    auto normal = [nx, ny, gb] __device__ (int i, int j) {
        int x = clamp(i, 0, nx-1);
        int y = clamp(j, 0, ny-1);
        int idx = (ny-y-1)*nx + x;
        float4 buff = gb[idx];
        return Vec3(buff.y, buff.z, buff.w);
    };

    auto depth = [nx, ny, gb] __device__ (int i, int j) {
        int x = clamp(i, 0, nx-1);
        int y = clamp(j, 0, ny-1);
        int idx = (ny-y-1)*nx + x;
        float4 buff = gb[idx];
        return buff.x;
    };

    int lw = 2; // Line width

    Vec3 center   = normal(i, j);
    Vec3 n  = normal(i,    j+lw);
    Vec3 s  = normal(i,    j-lw);
    Vec3 e  = normal(i+lw, j   );
    Vec3 w  = normal(i-lw, j   );
    Vec3 nw = normal(i-lw, j+lw);
    Vec3 ne = normal(i+lw, j+lw);
    Vec3 sw = normal(i-lw, j-lw);
    Vec3 se = normal(i+lw, j-lw);

    float dcenter =   depth(i, j);
    float dn  = depth(i,    j+lw);
    float ds  = depth(i,    j-lw);
    float de  = depth(i+lw, j   );
    float dw  = depth(i-lw, j   );
    float dnw = depth(i-lw, j+lw);
    float dne = depth(i+lw, j+lw);
    float dsw = depth(i-lw, j-lw);
    float dse = depth(i+lw, j-lw);

    Vec3 kernel0((nw-center).length(), (n -center).length(), (ne-center).length());
    Vec3 kernel1((w -center).length(),  0,                   (e -center).length());
    Vec3 kernel2((sw-center).length(), (s -center).length(), (se-center).length());

    float edge_x = dot(sobel_x0, kernel0) + dot(sobel_x1, kernel1) + dot(sobel_x2, kernel2);
    float edge_y = dot(sobel_y0, kernel0) + dot(sobel_y1, kernel1) + dot(sobel_y2, kernel2);
    float edge_normal = length(edge_x, edge_y);

    kernel0 = Vec3(dnw-dcenter, dn -dcenter, dne-dcenter);
    kernel1 = Vec3(dw -dcenter, 0          , de -dcenter);
    kernel2 = Vec3(dsw-dcenter, ds -dcenter, dse-dcenter);

    edge_x = dot(sobel_x0, kernel0) + dot(sobel_x1, kernel1) + dot(sobel_x2, kernel2);
    edge_y = dot(sobel_y0, kernel0) + dot(sobel_y1, kernel1) + dot(sobel_y2, kernel2);
    float edge_depth = length(edge_x, edge_y);

    float threshold = 1.0;
    float edge = max(edge_normal, edge_depth*0.2f);
    if (edge > threshold) {
        int pixel_index = (ny-j-1)*nx + i;
        fb[pixel_index] = Vec3(0);
    }
}

__global__ void fog(Vec3 *fb, float4 *gb, float max_depth, size_t w, size_t h) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= w) || (j >= h)) return;
    int pixel_index = (h-j-1)*w + i;

    Vec3 bg_color(1.0f, 0.98f, 0.92f);
    float a = smoothstep(max_depth*0.7f, max_depth, gb[pixel_index].x);
    fb[pixel_index] = (1-a)*fb[pixel_index] + a*bg_color;
}

__global__ void vignette(Vec3 *fb, size_t w, size_t h) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= w) || (j >= h)) return;
    int pixel_index = (h-j-1)*w + i;
    float u = float(i) / float(w);
    float v = float(j) / float(h);

    float r_edge = 0.65f;
    float r_pix  = length(u-0.5f, v-0.5f);
    float softness = 0.3f;
    float strength = 0.75f;

    float vignette = smoothstep(r_edge-softness, r_edge, r_pix)*strength;

    Vec3 col_top(0.90f, 0.64f, 0.55f);
    Vec3 col_bot(0.44f, 0.71f, 0.67f);
    Vec3 blend = (1.0f-v)*col_bot + v*col_top;

    Vec3 vignette_color = (1.0f-vignette) * blend;
    fb[pixel_index] = (1.0f-vignette)*fb[pixel_index] + vignette*vignette_color;
}

__global__ void downsample(uint8_t* out, Vec3 *in, size_t w, size_t h) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i_in = i*2;
    int j_in = j*2;
    if((i >= w/2) || (j >= h/2)) return;
    int idx = (h/2-j-1)*w/2 + i;
    
    int top_left  = (h-j_in-1)*w + i_in;
    int top_right = (h-j_in-1)*w + i_in + 1;
    int bot_left  = (h-j_in-2)*w + i_in;
    int bot_right = (h-j_in-2)*w + i_in + 1;

    const Vec3& tl = in[top_left ];
    const Vec3& tr = in[top_right];
    const Vec3& bl = in[bot_left ];
    const Vec3& br = in[bot_right];

    Vec3 color = lerp(lerp(tl, tr, 0.5f), lerp(bl, br, 0.5f), 0.5f);

    out[idx*3]   = color[0] * 255.99;
    out[idx*3+1] = color[1] * 255.99;
    out[idx*3+2] = color[2] * 255.99;
}
