#pragma once

#include "bbox.h"
#include "util/mathlib.h"

#include <cuda/cmath>

typedef unsigned char uchar;

enum Dim {
    X,
    Y,
    Z
};

struct Volume {
    cudaTextureObject_t tex;
    cudaExtent          dims;
    BBox                box;
};

struct HitRecord {
    float t;
    float3 p;
    float3 normal;
};
    
// Voxel traversal from Amantides and Woo
__device__ bool hitVolume(const Ray& r,
                          Volume v,
                          float tmin, 
                          float tmax,
                          HitRecord& rec) {
    // Intersection times stored in tmin and tmax
    float3 normal;
    if (!v.box.hitNormal(r, tmin, tmax, normal)) return false;

    if (tmin < 0) printf("whoops");

    // Initializing variables
    float3 pos = r.at(tmin + EPS_F);
    
    int x = clamp((int)pos.x, 0, v.dims.width), 
        y = clamp((int)pos.y, 0, v.dims.height), 
        z = clamp((int)pos.z, 0, v.dims.depth);
    
    int3 step(cuda::std::signbit(r.d.x) ? -1 : 1,
              cuda::std::signbit(r.d.y) ? -1 : 1,
              cuda::std::signbit(r.d.z) ? -1 : 1);
    
    float3 t_next(((float)(x + step.x) - r.o.x) / r.d.x,
                  ((float)(y + step.y) - r.o.y) / r.d.y,
                  ((float)(z + step.z) - r.o.z) / r.d.z);
    
    float3 t_delta(1.0f / abs(r.d.x),
                   1.0f / abs(r.d.y),
                   1.0f / abs(r.d.z));

    // If we exit the volume, which side will it be?
    int3 out(step.x > 0 ? (int)v.dims.width  : -1,
             step.y > 0 ? (int)v.dims.height : -1,
             step.z > 0 ? (int)v.dims.depth  : -1);

    uchar val = tex3D<uchar>(v.tex, (float)x, (float)y, (float)z);

    if (val != 0) {
        rec.t = tmin + EPS_F;
        rec.p = pos;
        rec.normal = normal;
        return true;
    }

    // Finding minimum t value that hits a voxel boundary and stepping in that direction
    Dim last_dim;
    while (val == 0) {
        if (t_next.x < t_next.y) {
            if (t_next.x < t_next.z) {
                x = x + step.x;
                if (x == out.x) return false;
                t_next.x = t_next.x + t_delta.x;
                last_dim = X;
            } else {
                z = z + step.z;
                if (z == out.z) return false;
                t_next.z = t_next.z + t_delta.z;
                last_dim = Z;
            }
        } else {
            if (t_next.y < t_next.z) {
                y = y + step.y;
                if (y == out.y) return false;
                t_next.y = t_next.y + t_delta.y;
                last_dim = Y;
            } else {
                z = z + step.z;
                if (z == out.z) return false;
                t_next.z = t_next.z + t_delta.z;
                last_dim = Z;
            }
        }
        val = tex3D<uchar>(v.tex, (float)x+0.5f, (float)y+0.5f, (float)z+0.5f);
    }

    float t_hit;
    switch (last_dim) {
        case X: 
            t_hit = t_next.x - t_delta.x; 
            normal = make_float3(-step.x, 0.0f, 0.0f);
            break;
        case Y: 
            t_hit = t_next.y - t_delta.y; 
            normal = make_float3(0.0f, -step.y, 0.0f);
            break;
        case Z: 
            t_hit = t_next.z - t_delta.z; 
            normal = make_float3(0.0f, 0.0f, -step.z);
            break;
    }

    rec.t = t_hit;
    rec.p = r.at(t_hit);
    rec.normal = normal;
    return true;
}