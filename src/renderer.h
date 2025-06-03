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
        __device__ Renderer(Image _out, float4 *_gb,
                            Hitable **_world, BBox **_world_bounds, curandState *_rand_state) : 
                            world(_world), 
                            out(_out),
                            gb(_gb),
                            world_bounds(_world_bounds), 
                            rand_state(_rand_state) {
            Vec3 cam_pos(60.0f, 42.0f, 78.0f);
            Vec3 cam_target(100.0f, -20.0f, 150.0f);
            Vec3 up(0, 1.0f, 0);
            float vfov = 90;
            float aspect = (float)out.w/(float)out.h;
            cam = Camera(cam_pos, cam_target, up, vfov, aspect);
        }
        
        __device__ HitRecord trace_ray(const Ray& r, float max_dist) {
            HitRecord rec;
            while (true) {
                if ((*world)->hit(r, EPS_F, FLT_MAX, rec)) {
                    if (rec.t + r.t_offset > max_dist) break;
                    rec.t += r.t_offset;
                    return rec;
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
            rec.t = max_dist;
            return rec;
        }

        __device__ Vec3 buff_to_color(const Ray& r, const HitRecord& buff, const float max_dist) {
            //return buff.normal*0.5f+0.5f;  // Normal colors (for debugging)
            Vec3 bg_col(1.0f, 0.98f, 0.92f);
            if (buff.t + EPS_F > max_dist) return bg_col;

            // Phong lighting
            Vec3 light_dir = Vec3(0.7f, 1.0f, 0.5f).unit();
            Vec3 view_dir = -r.dir();
            Vec3 refl_dir = reflect(-light_dir, buff.normal);

            // Warm for lit areas, cool for unlit
            Vec3 cool(0.4f, 0.65f, 0.61f);
            Vec3 warm(0.9f, 0.72f, 0.78f);
            Vec3 spec(0.2f, 0.18f, 0.19f);

            Ray shadow_ray(buff.p, -light_dir);
            float shadow_dist = (*world_bounds)->max.max();
            HitRecord shadow_buff = trace_ray(shadow_ray, shadow_dist);
            float occlusion = shadow_buff.t/shadow_dist;

            float t = dot(light_dir, buff.normal) * 0.5f + 0.5f;
            float s = fmaxf(0, dot(view_dir, refl_dir));
            s = powf(s, 20.0f);
            Vec3 diffuse = lerp(cool, warm, t);
            Vec3 specular = s * spec;
            Vec3 color = (diffuse+specular).clamp();

            float a = smoothstep(0, max_dist*0.5f, buff.t);
            float shadow_factor = lerp(0.4f, 1.0f, a);
            Vec3 shadowed = lerp(color*shadow_factor, color, occlusion);
            return lerp(shadowed, bg_col, a*0.7f);
        }

        Hitable **world;
        BBox **world_bounds;
        Image out;
        float4 *gb;
        Camera cam;
        curandState *rand_state;
};
