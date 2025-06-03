#pragma once

#include "vec3.h"

struct Ray {
    __device__ Ray() {};
    __device__ explicit Ray(const Vec3 &origin, const Vec3 &dir) : o(origin), d(dir.unit()), t_offset(0) {}
    __device__ Vec3 origin() const { return o; }
    __device__ Vec3 dir() const    { return d; }
    __device__ Vec3 at(const float t) const { return o + d*t; }
    __device__ Vec3 at_world(const float t) const { return at(t + t_offset); } 
    __device__ void wrap(const float t, const Vec3& s) const {
        const float eps = 1e-3f;
        o = at(t + eps);
        t_offset += t + eps;
        if (o.x >= s.x) o.x -= s.x;
        if (o.y >= s.y) o.y -= s.y;
        if (o.z >= s.z) o.z -= s.z;
        if (o.x <= 0)   o.x += s.x;
        if (o.y <= 0)   o.y += s.y;
        if (o.z <= 0)   o.z += s.z;
    }

    Vec3 d;
    mutable Vec3 o;
    mutable float t_offset;
};
