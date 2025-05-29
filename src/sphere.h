#pragma once

#include "hitable.h"

class Sphere: public Hitable {
    public:
        __device__ Sphere() {}
        __device__ Sphere(Vec3 c, float r) : center(c), radius(r) {}
        __device__ virtual bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const;
        Vec3 center;
        float radius; 
};

__device__ bool Sphere::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    Vec3 oc = r.origin() - center;
    float a = 1.0f;
    float b = 2.0f * dot(oc, r.dir());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - 4.0f*a*c;

    float d0, d1;
    if(discriminant < 0) return false;
    float q;
    float sqrt_discr = sqrtf(discriminant);
    if(b > 0) q = -0.5f * (b + sqrt_discr);
    else q = -0.5f * (b - sqrt_discr);
    d0 = q / a;
    d1 = c / q;
    if (d0 > d1) {
        float temp = d0;
        d0 = d1;
        d1 = temp;
    }

    if(d0 < t_min || d0 > t_max) {
        d0 = d1;
        if(d0 < t_min || d0 > t_max) return false;
    }

    rec.t = d0;
    rec.p = (r.origin() + (r.dir() * d0));
    rec.normal = (rec.p - center) / radius;
    return true;
}
