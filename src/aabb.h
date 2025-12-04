#ifndef AABBH
#define AABBH

#include <float.h>
#include "ray.h"


class AABB{
    public:
        __device__ AABB() {}
        __device__ AABB(vec3 mn, vec3 mx) : min(mn), max(mx) {};
        __device__ AABB(hitable** list, int n) {
            vec3 min_corner(FLT_MAX, FLT_MAX, FLT_MAX);
            vec3 max_corner(-FLT_MAX, -FLT_MAX, -FLT_MAX);

            for (int i = 0; i < n; i++) {
                AABB b = list[i]->bbox();
                min_corner.x() = fminf(min_corner.x(), b.min.x());
                min_corner.y() = fminf(min_corner.y(), b.min.y());
                min_corner.z() = fminf(min_corner.z(), b.min.z());

                max_corner.x() = fmaxf(max_corner.x(), b.max.x());
                max_corner.y() = fmaxf(max_corner.y(), b.max.y());
                max_corner.z() = fmaxf(max_corner.z(), b.max.z());
            }

            min = min_corner;
            max = max_corner;
        };
        __device__ AABB(AABB ab1, AABB ab2) {
            vec3 min_corner(FLT_MAX, FLT_MAX, FLT_MAX);
            vec3 max_corner(-FLT_MAX, -FLT_MAX, -FLT_MAX);

            min_corner.x() = fminf(ab1.min.x(), ab2.min.x());
            min_corner.y() = fminf(ab1.min.y(), ab2.min.y());
            min_corner.z() = fminf(ab1.min.z(), ab2.min.z());

            max_corner.x() = fmaxf(ab1.max.x(), ab2.max.x());
            max_corner.y() = fmaxf(ab1.max.y(), ab2.max.y());
            max_corner.z() = fmaxf(ab1.max.z(), ab2.max.z());

            min = min_corner;
            max = max_corner;
        }

        __device__ int get_longest_axis() const {
            float dx = max.x() - min.x();
            float dy = max.y() - min.y();
            float dz = max.z() - min.z();

            if (dx >= dy && dx >= dz) return 0;
            else if (dy >= dz) return 1;
            else return 2;
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
