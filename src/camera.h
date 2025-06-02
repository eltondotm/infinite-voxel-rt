#pragma once

#include "ray.h"

class Camera {
    public:
        __device__ Camera() {
            lower_left = Vec3(-2.0f, -1.0f, 0.0f);
            hori = Vec3(4.0f, 0.0f, 0.0f);
            vert = Vec3(0.0f, 2.0f, 0.0f);
            origin = Vec3(0.0f, 0.0f, -2.0f);
        }

        __device__ Ray generate_ray(float u, float v) {
            return Ray(origin, lower_left + u*hori + v*vert - origin);
        }
        
        Vec3 origin;
        Vec3 lower_left;
        Vec3 hori;
        Vec3 vert;
};