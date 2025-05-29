#pragma once

#include "vec3.h"

struct Ray {
    __device__ Ray() {};
    __device__ explicit Ray(const Vec3 &origin, const Vec3 &dir) : o(origin), d(dir.unit()) {}
    __device__ Vec3 origin() const { return o; }
    __device__ Vec3 dir() const    { return d; }
    __device__ Vec3 at(float t) const { return o + d*t; }

    Vec3 o;
    Vec3 d;
};
