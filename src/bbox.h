#pragma once

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "ray.h"
#include "vec3.h"
#include "util/device_util.h"

struct BBox {

    /// Default min is max float value, default max is negative max float value
    __device__ BBox() : min(FLT_MAX), max(-FLT_MAX) {
    }
    /// Set minimum and maximum extent
    __device__ explicit BBox(Vec3 min, Vec3 max) : min(min), max(max) {
    }

    /// Rest min to max float, max to negative max float
    __device__ void reset() {
        min = Vec3(FLT_MAX);
        max = Vec3(-FLT_MAX);
    }

    /// Expand bounding box to include point
    __device__ void enclose(Vec3 point) {
        min = hmin(min, point);
        max = hmax(max, point);
    }
    __device__ void enclose(BBox box) {
        min = hmin(min, box.min);
        max = hmax(max, box.max);
    }

    /// Get center point of box
    __device__ Vec3 center() const {
        return (min + max) * 0.5f;
    }

    // Check whether box has no volume
    __device__ bool empty() const {
        return min.x > max.x || min.y > max.y || min.z > max.z;
    }

    /// Get surface area of the box
    __device__ float surface_area() const {
        if(empty()) return 0.0f;
        Vec3 extent = max - min;
        return 2.0f * (extent.x * extent.z + extent.x * extent.y + extent.y * extent.z);
    }

    __device__ int hit(const Ray& r, float& t_min, float& t_max) const {
        int hit_dim = -1;

        // Intersect x planes
        float tx_min = (min.x - r.origin().x) / r.dir().x;
        float tx_max = (max.x - r.origin().x) / r.dir().x;
        if(tx_min > tx_max) swap(tx_min, tx_max);  // Depends on ray orientation
        if(tx_min > t_max || tx_max < t_min) return 0;  // No overlap
        if(tx_min > t_min) { t_min = tx_min; hit_dim = 1; }
        if(tx_max < t_max)   t_max = tx_max;

        // Intersect y planes
        float ty_min = (min.y - r.origin().y) / r.dir().y;
        float ty_max = (max.y - r.origin().y) / r.dir().y;
        if(ty_min > ty_max) swap(ty_min, ty_max);
        if(ty_min > t_max || ty_max < t_min) return 0;
        if(ty_min > t_min) { t_min = ty_min; hit_dim = 2; }
        if(ty_max < t_max)   t_max = ty_max;
        
        // Intersect z planes
        float tz_min = (min.z - r.origin().z) / r.dir().z;
        float tz_max = (max.z - r.origin().z) / r.dir().z;
        if(tz_min > tz_max) swap(tz_min, tz_max);
        if(tz_min > t_max || tz_max < t_min) return 0;
        if(tz_min > t_min) { t_min = tz_min; hit_dim = 3; }
        if(tz_max < t_max) t_max = tz_max;

        return hit_dim;
    }

    Vec3 min;
    Vec3 max;
};
