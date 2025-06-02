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
    int x = (int)pos.x, 
        y = (int)pos.y, 
        z = (int)pos.z;
    IntVec3 step(cuda::std::signbit(r.dir().x) ? -1 : 1,
                 cuda::std::signbit(r.dir().y) ? -1 : 1,
                 cuda::std::signbit(r.dir().z) ? -1 : 1);
    Vec3 t_next(init_tmax(r, 0, x + step.x),
                init_tmax(r, 1, y + step.y),
                init_tmax(r, 2, z + step.z));
    Vec3 t_delta(1.0f / r.dir().x,
                 1.0f / r.dir().y,
                 1.0f / r.dir().z);

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

    // Finding minimum t value that hits a voxel boundary and stepping in that direction
    size_t steps = 0;
    do {
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
        if (++steps > 3) printf("%d", last_dim);
        val = tex3D<VolumeType>(volume, (float)x, (float)y, (float)z);
    } while (val == 0);

    float t_hit = t_next[last_dim] - t_delta[last_dim];
    Vec3 n(0);
    n[last_dim] = (float)-step.data[last_dim];

    rec.t = t_hit;
    rec.p = r.at(t_hit);
    rec.normal = n;
    return true;
}

__forceinline__ __device__ float Volume::init_tmax(const Ray& r, int dim, int idx) const {
    float distance = (float)(idx) - r.origin()[dim];
    return distance / r.dir()[dim];
}
