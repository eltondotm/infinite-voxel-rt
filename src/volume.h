#pragma once

#include <cuda/cmath>

#include "hitable.h"
#include "util/error.h"

typedef unsigned char VolumeType;


class Volume: public Hitable {
    public:
        __device__ Volume(cudaTextureObject_t v, cudaExtent d) : volume(v), dims(d) {
            Vec3 min_coord(0);
            Vec3 max_coord(dims.width, dims.height, dims.depth);
            bbox = BBox(min_coord, max_coord);
        }
        __device__ virtual bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const;

        BBox bbox;
        cudaTextureObject_t volume;
        cudaExtent dims;
    private:
        __forceinline__ __device__ float init_tmax(const Ray& r, int dim, int next_coord) const;
};

// Voxel traversal from Amantides and Woo
__device__ bool Volume::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    // Intersection times stored in t_min and t_max
    int last_dim = bbox.hit(r, t_min, t_max) - 1;
    if (last_dim == -1) return false;

    // Initializing variables
    Vec3 pos = r.at(t_min + EPS_F);
    int x = clamp((int)pos.x, 0, dims.width), 
        y = clamp((int)pos.y, 0, dims.height), 
        z = clamp((int)pos.z, 0, dims.depth);
    IntVec3 step(cuda::std::signbit(r.dir().x) ? -1 : 1,
                 cuda::std::signbit(r.dir().y) ? -1 : 1,
                 cuda::std::signbit(r.dir().z) ? -1 : 1);
    Vec3 t_next(init_tmax(r, 0, x + step.x),
                init_tmax(r, 1, y + step.y),
                init_tmax(r, 2, z + step.z));
    Vec3 t_delta(1.0f / abs(r.dir().x),
                 1.0f / abs(r.dir().y),
                 1.0f / abs(r.dir().z));

    // If we exit the volume, which side will it be?
    IntVec3 out(step.x > 0 ? (int)dims.width  : -1,
                step.y > 0 ? (int)dims.height : -1,
                step.z > 0 ? (int)dims.depth  : -1);

    VolumeType val = tex3D<VolumeType>(volume, (float)x, (float)y, (float)z);

    if (val != 0) {
        Vec3 n(0);
        n[last_dim] = (float)-step.data[last_dim];

        rec.t = t_min;
        rec.p = r.at(t_min);
        rec.normal = n;
        return true;
    }

    // Fixed step size traversal
    if (false) {
        float t = t_min;
        float t_step = 0.001f;
        Vec3 p_step = r.dir()*t_step;
        Vec3 pos = r.at(t);

        while (val == 0) {
            pos += p_step;
            t += t_step;
            if(!bbox.contains(pos)) return false;
            val = tex3D<VolumeType>(volume, pos.x, pos.y, pos.z);
        }

        Vec3 pos_prev = pos - p_step;
        if ((int)pos_prev.x != (int)pos.x) last_dim = 0;
        if ((int)pos_prev.y != (int)pos.y) last_dim = 1;
        if ((int)pos_prev.z != (int)pos.z) last_dim = 2;
        Vec3 n(0);
        n[last_dim] = (float)(-step.data[last_dim]);

        rec.t = t;
        rec.p = r.at(t);
        rec.normal = n;
        return true;
    }

    // Finding minimum t value that hits a voxel boundary and stepping in that direction
    while (val == 0) {
        if (t_next.x < t_next.y) {
            if (t_next.x < t_next.z) {
                x = x + step.x;
                if (x == out.x) return false;
                t_next.x = t_next.x + t_delta.x;
                last_dim = 0;
            } else {
                z = z + step.z;
                if (z == out.z) return false;
                t_next.z = t_next.z + t_delta.z;
                last_dim = 2;
            }
        } else {
            if (t_next.y < t_next.z) {
                y = y + step.y;
                if (y == out.y) return false;
                t_next.y = t_next.y + t_delta.y;
                last_dim = 1;
            } else {
                z = z + step.z;
                if (z == out.z) return false;
                t_next.z = t_next.z + t_delta.z;
                last_dim = 2;
            }
        }
        val = tex3D<VolumeType>(volume, (float)x+0.5f, (float)y+0.5f, (float)z+0.5f);
    }

    float t_hit = t_next[last_dim] - t_delta[last_dim];
    Vec3 n(0);
    n[last_dim] = -(float)step.data[last_dim];

    rec.t = t_hit;
    rec.p = r.at(t_hit);
    rec.normal = n;
    return true;
}

__forceinline__ __device__ float Volume::init_tmax(const Ray& r, int dim, int idx) const {
    float distance = (float)(idx) - r.origin()[dim];
    return distance / r.dir()[dim];
}
