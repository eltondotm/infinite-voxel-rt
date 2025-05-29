#pragma once

#include "util/error.h"
#include "util/random.h"

#include "hitable.h"
#include "camera.h"

#define N_COPIES 5

struct Image {
    __device__ Image(Vec3 *_fb, int _w, int _h) : fb(_fb), w(_w), h(_h) {}

    Vec3 *fb;
    int w;
    int h;
};

class Renderer {
    public:
        __device__ Renderer(Image _out, Hitable **_world, curandState *_rand_state) : 
            world(_world), out(_out), rand_state(_rand_state) {
            cam = Camera();
        }
        
        __device__ Vec3 trace_ray(const Ray& r) {
            HitRecord rec;
            for (int i = 0; i < N_COPIES; ++i) {
                if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
                    Vec3 light_dir = Vec3(1.0f, 1.0f, 0.0f).unit();
                    Vec3 view_dir = (cam.origin - rec.p).unit();
                    Vec3 refl_dir = reflect(-light_dir, rec.normal);

                    Vec3 cool(0.4f, 0.4f, 0.5f);
                    Vec3 warm(0.8f, 0.6f, 0.7f);
                    Vec3 spec(1.0f, 0.8f, 0.9f);

                    float t = dot(light_dir, rec.normal) * 0.5f + 0.5f;
                    float s = fmaxf(0, dot(view_dir, refl_dir));
                    s = (s * s) * (s * s) * (s * s) * s;
                    Vec3 diffuse = (1-t)*cool + t*warm;
                    Vec3 specular = s * spec;

                    return (diffuse + specular).clamp();
                }
            }
            float t = 0.5f*(r.dir().y + 1.0f);
            return (1.0f-t)*Vec3(1.0, 1.0, 1.0) + t*Vec3(0.5, 0.7, 1.0); 
        }

        Hitable **world;
        Image out;
        Camera cam;
        curandState *rand_state;
};
