#ifndef RAY_HALFH
#define RAY_HALFH
#include "vec3_half.h"

class ray_half
{
    public:
        __device__ ray_half() {}
        __device__ ray_half(const vec3_half& a, const vec3_half& b) { A = a; B = b; }
        __device__ vec3_half origin() const       { return A; }
        __device__ vec3_half direction() const    { return B; }
        __device__ vec3_half point_at_parameter(half t) const { return A + t*B; }

        vec3_half A;
        vec3_half B;
};

#endif
