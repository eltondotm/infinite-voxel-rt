#pragma once

#include "vec3.h"
#include "util/device_util.h"

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
