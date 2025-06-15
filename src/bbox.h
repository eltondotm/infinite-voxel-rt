#pragma once

#include "ray.h"

#include "util/mathlib.h"

struct BBox {
    // Default: unit box
    __device__ BBox() { min = make_float3(-1.0f); max = make_float3(1.0f); }
    __device__ BBox(float3 _min, float3 _max) : min(_min), max(_max) {}

    __device__ float3 extent() const {
        return max - min;
    }
    
    __device__ float3 center() const {
        return (max + min) * 0.5f;
    }

    __device__ bool hit(const Ray r, float& tnear, float& tfar) const {
        // Intersect x planes
        float tx_min = (min.x - r.o.x) / r.d.x;
        float tx_max = (max.x - r.o.x) / r.d.x;
        if(tx_min > tx_max) swap(tx_min, tx_max);
        if(tx_min > tfar || tx_max < tnear) return false;  // No overlap
        if(tx_min > tnear) tnear = tx_min;
        if(tx_max < tfar) tfar = tx_max;

        // Intersect y planes
        float ty_min = (min.y - r.o.y) / r.d.y;
        float ty_max = (max.y - r.o.y) / r.d.y;
        if(ty_min > ty_max) swap(ty_min, ty_max);
        if(ty_min > tfar || ty_max < tnear) return false;
        if(ty_min > tnear) tnear = ty_min;
        if(ty_max < tfar) tfar = ty_max;
        
        // Intersect z planes
        float tz_min = (min.z - r.o.z) / r.d.z;
        float tz_max = (max.z - r.o.z) / r.d.z;
        if(tz_min > tz_max) swap(tz_min, tz_max);
        if(tz_min > tfar || tz_max < tnear) return false;
        if(tz_min > tnear) tnear = tz_min;
        if(tz_max < tfar) tfar = tz_max;

        return true;
    }

    __device__ bool hitNormal(Ray r, float& tnear, float& tfar, float3& n) const {
        if (!hit(r, tnear, tfar)) return false;

        float3 intersection = r.at(tnear);

        // Transform to unit box
        intersection -= center();
        intersection *= 2.0f / extent();

        // Truncating to integer component
        intersection.x = (float)((int)(intersection.x + EPS_F));
        intersection.y = (float)((int)(intersection.y + EPS_F));
        intersection.z = (float)((int)(intersection.z + EPS_F));
        
        // Normalize in case of edges or corners
        n = normalize(intersection);

        return true;
    }

    float3 min;
    float3 max;
};
