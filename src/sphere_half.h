#ifndef SPHERE_HALFH
#define SPHERE_HALFH

#include "hitable_half.h"

class sphere_half : public hitable_half {
public:
    __device__ sphere_half() {}
    __device__ sphere_half(vec3_half cen, half r, material_half* m) :
        radius(r), mat_ptr(m)
#ifndef BVH
        , _center(cen)
#endif
    {
#ifdef BVH
        _center = cen;
        _bbox = AABB_half(cen - vec3_half(radius, radius, radius), cen + vec3_half(radius, radius, radius));
#endif
    }
    __device__ virtual bool hit(const ray_half& r, half tmin, half tmax, hit_record_half& rec) const;
    half radius;
    material_half* mat_ptr;
#ifndef BVH
    vec3_half _center;
#endif
};

__device__ bool sphere_half::hit(const ray_half& r, half t_min, half t_max, hit_record_half& rec) const {
    // Convert inputs to float for stable geometric math
    float3 oc_f = {__half2float(r.origin().x() - _center.x()), 
                   __half2float(r.origin().y() - _center.y()), 
                   __half2float(r.origin().z() - _center.z())};
    float3 dir_f = {__half2float(r.direction().x()), 
                    __half2float(r.direction().y()), 
                    __half2float(r.direction().z())};
    float radius_f = __half2float(radius);

    // Perform the quadratic formula in float32 (mixed precision)
    float a = dir_f.x*dir_f.x + dir_f.y*dir_f.y + dir_f.z*dir_f.z;
    float b = oc_f.x*dir_f.x + oc_f.y*dir_f.y + oc_f.z*dir_f.z;
    float c = (oc_f.x*oc_f.x + oc_f.y*oc_f.y + oc_f.z*oc_f.z) - radius_f*radius_f;
    float discriminant = b*b - a*c;

    if (discriminant > 0.0f) {
        float sqrt_disc = sqrtf(discriminant);
        float t_min_f = __half2float(t_min);
        float t_max_f = __half2float(t_max);

        float temp = (-b - sqrt_disc) / a;
        if (temp < t_max_f && temp > t_min_f) {
            rec.t = __float2half(temp);
            rec.p = r.point_at_parameter(rec.t);
            float3 p_f = {__half2float(rec.p.x()), __half2float(rec.p.y()), __half2float(rec.p.z())};
            float3 c_f = {__half2float(_center.x()), __half2float(_center.y()), __half2float(_center.z())};
            float inv_r = 1.0f / __half2float(radius);
            rec.normal = vec3_half(
                __float2half((p_f.x - c_f.x) * inv_r),
                __float2half((p_f.y - c_f.y) * inv_r),
                __float2half((p_f.z - c_f.z) * inv_r)
            );
            
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}
#endif
