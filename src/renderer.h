#pragma once

#include "volume.h"

#include <cuda_runtime.h>

struct Scene {
    Volume volume;
    BBox bounds;
};

__device__ HitRecord traceRay(Ray r, const Scene* scene, float maxDist) {
    HitRecord rec;
    while (true) {
        if (hitVolume(r, scene->volume, EPS_F, FLT_MAX, rec)) {
            if (rec.t + r.t > maxDist) break;
            rec.t += r.t;
            return rec;
        } else {
            float t_min = EPS_F, t_max = FLT_MAX;
            if (scene->bounds.hit(r, t_min, t_max)) {
                r.wrap(t_max, scene->bounds.min, scene->bounds.max);
            } else {
                break;
            }
            if (r.t > maxDist) break;
        }
    }
    rec.t = maxDist;
    return rec;
}

__device__ float3 buffToColor(const Ray& r, const HitRecord& buff, const Scene* scene, float maxDist) {
    float3 bg_col = make_float3(1.0f, 0.98f, 0.92f);
    if (buff.t + EPS_F > maxDist) return bg_col;

    // Phong lighting
    float3 light_dir = normalize(make_float3(0.7f, -1.0f, 0.5f));
    float3 view_dir = r.d;
    float3 refl_dir = reflect(-light_dir, buff.normal);

    // Warm for lit areas, cool for unlit
    float3 cool = make_float3(0.4f, 0.65f, 0.61f);
    float3 warm = make_float3(0.9f, 0.72f, 0.78f);
    float3 spec = make_float3(0.2f, 0.18f, 0.19f);

    Ray shadow_ray(buff.p, -light_dir);
    float shadow_dist = scene->bounds.extent().y;
    HitRecord shadow_buff = traceRay(shadow_ray, scene, shadow_dist);
    float occlusion = shadow_buff.t/shadow_dist;

    float t = dot(-light_dir, buff.normal) * 0.5f + 0.5f;
    float s = fmaxf(0, dot(view_dir, refl_dir));
    s = powf(s, 20.0f);
    float3 diffuse = lerp(cool, warm, t);
    float3 specular = s * spec;
    float3 color = clamp((diffuse+specular), 0.0f, 1.0f);

    float a = smoothstep(0, maxDist*0.5f, buff.t);
    float shadow_factor = lerp(0.4f, 1.0f, a);
    float3 shadowed = lerp(color*shadow_factor, color, occlusion);
    return lerp(shadowed, bg_col, a*0.7f);
}