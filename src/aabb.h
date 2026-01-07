#ifndef AABBH
#define AABBH

#ifdef BVH

#include <float.h>
#include "ray.h"

class AABB {
public:
    __device__ AABB() : AABB(vec3(FLT_MAX, FLT_MAX, FLT_MAX), vec3(FLT_MIN, FLT_MIN, FLT_MIN)) {}
    __device__ AABB(AABB ab1, AABB ab2) : AABB(ab1) { merge_with(ab2); }
    __device__ AABB(const AABB& other) { bounds[0] = other.bounds[0]; bounds[1] = other.bounds[1]; }
    __device__ AABB(vec3 min, vec3 max) { bounds[0] = min;  bounds[1] = max; }

    __device__ void merge_with(const AABB& other) {
        bounds[0].set_x(fminf(bounds[0].x(), other.bounds[0].x()));
        bounds[0].set_y(fminf(bounds[0].y(), other.bounds[0].y()));
        bounds[0].set_z(fminf(bounds[0].z(), other.bounds[0].z()));

        bounds[1].set_x(fmaxf(bounds[1].x(), other.bounds[1].x()));
        bounds[1].set_y(fmaxf(bounds[1].y(), other.bounds[1].y()));
        bounds[1].set_z(fmaxf(bounds[1].z(), other.bounds[1].z()));
    }

    __device__ int get_longest_axis() const {
        float dx = bounds[1].x() - bounds[0].x();
        float dy = bounds[1].y() - bounds[0].y();
        float dz = bounds[1].z() - bounds[0].z();

        if (dx >= dy && dx >= dz) return 0;
        if (dy >= dz) return 1;
        return 2;
    }

    __device__ virtual bool hit(const ray& r) const;
    vec3 bounds[2];
};

__device__ bool AABB::hit(const ray& r) const {
    vec3 invdir = vec3(1, 1, 1) / r.direction();
    float tmin, tmax, tymin, tymax, tzmin, tzmax;
    int sign[3] = {
        (invdir.x() < 0),
        (invdir.y() < 0),
        (invdir.z() < 0),
    };

    tmin = (bounds[sign[0]].x() - r.origin().x()) * invdir.x();
    tmax = (bounds[1 - sign[0]].x() - r.origin().x()) * invdir.x();
    tymin = (bounds[sign[1]].y() - r.origin().y()) * invdir.y();
    tymax = (bounds[1 - sign[1]].y() - r.origin().y()) * invdir.y();

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    tzmin = (bounds[sign[2]].z() - r.origin().z()) * invdir.z();
    tzmax = (bounds[1 - sign[2]].z() - r.origin().z()) * invdir.z();

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;

    return true;
}
#endif
#endif
