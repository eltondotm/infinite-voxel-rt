#pragma once

#include "util/error.h"
#include "util/random.h"

#include "hitable.h"
#include "bbox.h"
#include "camera.h"

struct Image {
    __device__ Image(Vec3 *_fb, int _w, int _h) : fb(_fb), w(_w), h(_h) {}

    Vec3 *fb;
    int w;
    int h;
};

class Renderer {
    public:
        __device__ Renderer(Image _out, Hitable **_world, BBox **_world_bounds, curandState *_rand_state) : 
            world(_world), out(_out), world_bounds(_world_bounds), rand_state(_rand_state) {
            Vec3 cam_pos(5.0f, 2.0f, 5.0f);
            Vec3 cam_target(4.0f, 0, 0);
            Vec3 up(0, 1.0f, 0);
            float vfov = 90;
            float aspect = (float)out.w/(float)out.h;
            cam = Camera(cam_pos, cam_target, up, vfov, aspect);
        }
        
        __device__ Vec3 trace_ray(const Ray& r, float max_dist) {
            Vec3 bg_col(1.0f, 0.98f, 0.92f);
            HitRecord rec;
            while (true) {
                if ((*world)->hit(r, EPS_F, FLT_MAX, rec)) {
                    if (rec.t + r.t_offset > max_dist) break;

                    // Phong lighting
                    Vec3 light_dir = Vec3(1.0f, 7.0f, 0.3f).unit();
                    Vec3 view_dir = -r.dir();
                    Vec3 refl_dir = reflect(-light_dir, rec.normal);

                    // Warm for lit areas, cool for unlit
                    Vec3 cool(0.4f, 0.4f, 0.5f);
                    Vec3 warm(0.8f, 0.6f, 0.7f);
                    Vec3 spec(1.0f, 0.8f, 0.9f);

                    float t = dot(light_dir, rec.normal) * 0.5f + 0.5f;
                    float s = fmaxf(0, dot(view_dir, refl_dir));
                    s = (s * s) * (s * s) * (s * s) * s;
                    Vec3 diffuse = (1-t)*cool + t*warm;
                    Vec3 specular = s * spec;
                    Vec3 color = (diffuse + specular).clamp();

                    float a = smoothstep(max_dist - 5.0f, max_dist, rec.t + r.t_offset);
                    return (1-a)*color + a*bg_col;
                } else {
                    float t_min = EPS_F, t_max = FLT_MAX;
                    if ((*world_bounds)->hit(r, t_min, t_max)) {
                        r.wrap(t_max, (*world_bounds)->max);
                    } else {
                        break;
                    }
                    if (r.t_offset > max_dist) break;
                }
            }
            float t = 0.5f*(r.dir().y + 1.0f);
            return bg_col; 
        }

        Hitable **world;
        BBox **world_bounds;
        Image out;
        Camera cam;
        curandState *rand_state;
};
