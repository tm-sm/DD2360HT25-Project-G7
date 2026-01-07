#ifndef HITABLE_HALFH
#define HITABLE_HALFH

#include "ray_half.h"
#include "aabb_half.h"

class material_half;

struct hit_record_half
{
    half t;
    vec3_half p;
    vec3_half normal;
    material_half* mat_ptr;
};

class hitable_half {
public:
    __device__ virtual bool hit(const ray_half& r, half t_min, half t_max, hit_record_half& rec) const = 0;
#ifdef BVH
    __device__ vec3_half center() const { return _center; }
    __device__ AABB_half bbox() const { return _bbox; }
protected:
    AABB_half _bbox;
    vec3_half _center;
#endif
};

#endif
