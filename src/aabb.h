#ifndef AABBH
#define AABBH

#ifdef BVH

#include <float.h>
#include "ray.h"

class AABB {
public:
    __device__ AABB() :min(0, 0, 0), max(0, 0, 0) {}
    __device__ AABB(vec3 min, vec3 max) : min(min), max(max) {};
    __device__ AABB(const AABB& other) : min(other.min), max(other.max) {}
    __device__ AABB(AABB ab1, AABB ab2) : AABB(ab1) {
        merge_with(ab2);
    }

    __device__ void merge_with(const AABB& other) {
        min.set_x(fminf(min.x(), other.min.x()));
        min.set_y(fminf(min.y(), other.min.y()));
        min.set_z(fminf(min.z(), other.min.z()));

        max.set_x(fmaxf(max.x(), other.max.x()));
        max.set_y(fmaxf(max.y(), other.max.y()));
        max.set_z(fmaxf(max.z(), other.max.z()));
    }

    __device__ int get_longest_axis() const {
        float dx = max.x() - min.x();
        float dy = max.y() - min.y();
        float dz = max.z() - min.z();

        if (dx >= dy && dx >= dz) return 0;
        if (dy >= dz) return 1;
        return 2;
    }

    __device__ virtual bool hit(const ray& r, float tmin, float tmax) const;
    vec3 min;
    vec3 max;
};

__device__ bool AABB::hit(const ray& r, float t_min, float t_max) const {
    for (int a = 0; a < 3; a++) {
        float invD = 1.0f / r.direction()[a];
        float t0 = (min[a] - r.origin()[a]) * invD;
        float t1 = (max[a] - r.origin()[a]) * invD;
        if (invD < 0.0f) {
            float temp = t0;
            t0 = t1;
            t1 = temp;
        }
        t_min = fmaxf(t0, t_min);
        t_max = fminf(t1, t_max);
        if (t_max <= t_min)
            return false;
    }
    return true;
}


#endif
#endif
