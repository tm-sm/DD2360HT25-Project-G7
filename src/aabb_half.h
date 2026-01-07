#ifndef AABB_HALFH
#define AABB_HALFH

#ifdef BVH

#include <cuda_fp16.h>
#include "ray_half.h"

class AABB_half {
public:
    __device__ AABB_half() : AABB_half(vec3_half(__float2half(FLT_MAX), __float2half(FLT_MAX), __float2half(FLT_MAX)), vec3_half(__float2half(FLT_MIN), __float2half(FLT_MIN), __float2half(FLT_MIN))) {}
    __device__ AABB_half(AABB_half ab1, AABB_half ab2) : AABB_half(ab1) { merge_with(ab2); }
    __device__ AABB_half(const AABB_half& other) { bounds[0] = other.bounds[0]; bounds[1] = other.bounds[1]; }
    __device__ AABB_half(vec3_half min, vec3_half max) { bounds[0] = min;  bounds[1] = max; }

    __device__ void merge_with(const AABB_half& other) {
        bounds[0].set_x(fminf(__half2float(bounds[0].x()), __half2float(other.bounds[0].x())));
        bounds[0].set_y(fminf(__half2float(bounds[0].y()), __half2float(other.bounds[0].y())));
        bounds[0].set_z(fminf(__half2float(bounds[0].z()), __half2float(other.bounds[0].z())));

        bounds[1].set_x(fmaxf(__half2float(bounds[1].x()), __half2float(other.bounds[1].x())));
        bounds[1].set_y(fmaxf(__half2float(bounds[1].y()), __half2float(other.bounds[1].y())));
        bounds[1].set_z(fmaxf(__half2float(bounds[1].z()), __half2float(other.bounds[1].z())));
    }

    __device__ int get_longest_axis() const {
        half dx = bounds[1].x() - bounds[0].x();
        half dy = bounds[1].y() - bounds[0].y();
        half dz = bounds[1].z() - bounds[0].z();

        if (dx >= dy && dx >= dz) return 0;
        if (dy >= dz) return 1;
        return 2;
    }

    __device__ virtual bool hit(const ray_half& r) const;
    vec3_half bounds[2];
};

__device__ bool AABB_half::hit(const ray_half& r) const {
    vec3_half invdir = vec3_half(__float2half(1.0f), __float2half(1.0f), __float2half(1.0f)) / r.direction();
    half tmin, tmax, tymin, tymax, tzmin, tzmax;
    int sign[3] = {
        (invdir.x() < __float2half(0.0f)),
        (invdir.y() < __float2half(0.0f)),
        (invdir.z() < __float2half(0.0f)),
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

    return true;
}
#endif
#endif
