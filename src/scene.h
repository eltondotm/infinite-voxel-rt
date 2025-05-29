#pragma once

#include "hitable.h"

class Scene: public Hitable {
    public:
        __device__ Scene() {}
        __device__ Scene(Hitable **l, int n) : list(l), size(n) {}
        __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const;
        Hitable **list;
        int size;
};

__device__ bool Scene::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    HitRecord temp;
    bool hit = false;
    float closest = t_max;
    for (int i = 0; i < size; ++i) {
        if (list[i]->hit(r, t_min, closest, temp)) {
            hit = true;
            closest = temp.t;
            rec = temp;
        }
    }
    return hit;
}
