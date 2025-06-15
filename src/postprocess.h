#pragma once

#include "util/mathlib.h"

#include <cuda_runtime.h>

__global__ void d_edgeDetect(uint *fb, float4 *gb, size_t nx, size_t ny) {
    const float3 sobel_x0(1.0f, 0.0f, -1.0f);
    const float3 sobel_x1(2.0f, 0.0f, -2.0f);
    const float3 sobel_x2(1.0f, 0.0f, -1.0f);

    const float3 sobel_y0(1.0f,   2.0f,  1.0f);
    const float3 sobel_y1(0.0f,   0.0f,  0.0f);
    const float3 sobel_y2(-1.0f, -2.0f, -1.0f);

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= nx) || (j >= ny)) return;

    auto normal = [nx, ny, gb] __device__ (int i, int j) {
        int x = clamp(i, 0, nx-1);
        int y = clamp(j, 0, ny-1);
        int idx = (ny-y-1)*nx + x;
        float4 buff = gb[idx];
        return float3(buff.x, buff.y, buff.z);
    };

    auto depth = [nx, ny, gb] __device__ (int i, int j) {
        int x = clamp(i, 0, nx-1);
        int y = clamp(j, 0, ny-1);
        int idx = (ny-y-1)*nx + x;
        float4 buff = gb[idx];
        return max(log2f(buff.w), 0.0f);
    };

    int lw = 1; // Line width

    float3 center   = normal(i, j);
    float3 n  = normal(i,    j+lw);
    float3 s  = normal(i,    j-lw);
    float3 e  = normal(i+lw, j   );
    float3 w  = normal(i-lw, j   );
    float3 nw = normal(i-lw, j+lw);
    float3 ne = normal(i+lw, j+lw);
    float3 sw = normal(i-lw, j-lw);
    float3 se = normal(i+lw, j-lw);

    float dcenter =   depth(i, j);
    float dn  = depth(i,    j+lw);
    float ds  = depth(i,    j-lw);
    float de  = depth(i+lw, j   );
    float dw  = depth(i-lw, j   );
    float dnw = depth(i-lw, j+lw);
    float dne = depth(i+lw, j+lw);
    float dsw = depth(i-lw, j-lw);
    float dse = depth(i+lw, j-lw);

    // In case there are issues with edge detecting small geometry
    //if (dcenter > 8.0f) return;

    float3 kernel0(length(nw-center), length(n -center), length(ne-center));
    float3 kernel1(length(w -center),                 0, length(e -center));
    float3 kernel2(length(sw-center), length(s -center), length(se-center));

    float edge_x = dot(sobel_x0, kernel0) + dot(sobel_x1, kernel1) + dot(sobel_x2, kernel2);
    float edge_y = dot(sobel_y0, kernel0) + dot(sobel_y1, kernel1) + dot(sobel_y2, kernel2);
    float edge_normal = length(make_float2(edge_x, edge_y));

    kernel0 = float3(dnw-dcenter, dn -dcenter, dne-dcenter);
    kernel1 = float3(dw -dcenter,           0, de -dcenter);
    kernel2 = float3(dsw-dcenter, ds -dcenter, dse-dcenter);

    edge_x = dot(sobel_x0, kernel0) + dot(sobel_x1, kernel1) + dot(sobel_x2, kernel2);
    edge_y = dot(sobel_y0, kernel0) + dot(sobel_y1, kernel1) + dot(sobel_y2, kernel2);
    float edge_depth = length(make_float2(edge_x, edge_y));

    float threshold = 0.5;
    float edge = max(edge_normal, edge_depth*2);
    if (edge > threshold) {
        int pixel_index = (ny-j-1)*nx + i;
        fb[pixel_index] = rgbaFloatToInt(make_float4(0.0f, 0.0f, 0.0f, 1.0f));
    }
}

__global__ void d_fog(uint *fb, float4 *gb, float max_depth, size_t w, size_t h) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= w) || (j >= h)) return;
    int pixel_index = (h-j-1)*w + i;

    float3 curr_color = make_float3(rgbaIntToFloat(fb[pixel_index]));

    float3 bg_color(1.0f, 0.98f, 0.92f);
    float a = smoothstep(max_depth*0.7f, max_depth, gb[pixel_index].w);
    fb[pixel_index] = rgbaFloatToInt(make_float4((1-a)*curr_color + a*bg_color, 1.0f));
}

__global__ void d_vignette(uint *fb, size_t w, size_t h) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= w) || (j >= h)) return;
    int pixel_index = (h-j-1)*w + i;
    float u = float(i) / float(w);
    float v = float(j) / float(h);

    float3 curr_color = make_float3(rgbaIntToFloat(fb[pixel_index]));

    float r_edge = 0.65f;
    float r_pix  = length(make_float2(u-0.5f, v-0.5f));
    float softness = 0.3f;
    float strength = 0.75f;

    float vignette = smoothstep(r_edge-softness, r_edge, r_pix)*strength;

    float3 col_top(0.90f, 0.64f, 0.55f);
    float3 col_bot(0.44f, 0.71f, 0.67f);
    float3 blend = (1.0f-v)*col_bot + v*col_top;

    float3 vignette_color = (1.0f-vignette) * blend;
    float3 blended_color  = (1.0f-vignette)*curr_color + vignette*vignette_color;
    fb[pixel_index] = rgbaFloatToInt(make_float4(blended_color, 1.0f));
}

__global__ void d_downsample(uint8_t* out, float3 *in, size_t w, size_t h) {
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

    const float3& tl = in[top_left ];
    const float3& tr = in[top_right];
    const float3& bl = in[bot_left ];
    const float3& br = in[bot_right];

    float3 color = lerp(lerp(tl, tr, 0.5f), lerp(bl, br, 0.5f), 0.5f);

    out[idx*3]   = color.x * 255.99;
    out[idx*3+1] = color.y * 255.99;
    out[idx*3+2] = color.z * 255.99;
}
