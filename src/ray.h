#pragma once

#include "util/mathlib.h"

struct BBox;

struct Ray {
    __device__ Ray(float3 _o, float3 _d) : o(_o), d(_d), t(0) {}

    __device__ float3 at(float time) const {
        return o + d*time;
    }

    __device__ void wrap(float time, float3 min, float3 max) {
        o = at(time + EPS_F);
        t += time + EPS_F;

        float3 dims = max - min;
        if (o.x >= max.x) o.x -= dims.x;
        if (o.y >= max.y) o.y -= dims.y;
        if (o.z >= max.z) o.z -= dims.z;
        if (o.x <= min.x) o.x += dims.x;
        if (o.y <= min.y) o.y += dims.y;
        if (o.z <= min.z) o.z += dims.z;
    }

    float3 o;
    float3 d;
    float  t;
};