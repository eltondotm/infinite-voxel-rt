#pragma once

#include "ray.h"

#define PI_F 3.1415926

class Camera {
    public:
        __device__ Camera() {}
        
        // Credit: Raytracing in a Weekend
        __device__ Camera(Vec3 pos, Vec3 target, Vec3 up, float vfov, float aspect) {
            Vec3 u, v, w;
            float theta = vfov*PI_F/180.0f;
            float half_height = tan(theta/2.0f);
            float half_width = aspect * half_height;
            w = (pos - target).unit();
            u = cross(up, w).unit();
            v = cross(w, u);

            origin = pos;
            lower_left = origin - half_width*u - half_height*v - w;
            hori = 2.0f*half_width*u;
            vert = 2.0f*half_height*v;
        }

        __device__ Ray generate_ray(float u, float v) {
            return Ray(origin, lower_left + u*hori + v*vert - origin);
        }
        
        Vec3 origin;
        Vec3 lower_left;
        Vec3 hori;
        Vec3 vert;
};