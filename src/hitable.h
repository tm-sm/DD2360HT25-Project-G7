#ifndef HITABLEH
#define HITABLEH

#include "ray.h"
#include "aabb.h"

class material;

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    material* mat_ptr;
};

class hitable {
public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
#ifdef BVH
    __device__ vec3 center() const { return _center; }
    __device__ AABB bbox() const { return _bbox; }
protected:
    AABB _bbox;
    vec3 _center;
#endif
};

#endif
