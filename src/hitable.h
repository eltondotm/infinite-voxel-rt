#pragma once

#include "ray.h"

struct HitRecord {
    float t;
    Vec3 p;
    Vec3 normal;
};

class Hitable {
    public:
        __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
};
