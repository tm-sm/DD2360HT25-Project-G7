#ifndef CAMERA_HALFH
#define CAMERA_HALFH

#include <curand_kernel.h>
#include "ray_half.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ vec3_half random_in_unit_disk_half(curandState *local_rand_state) {
    vec3_half p;
    do {
        p = __float2half(2.0f)*vec3_half(__float2half(curand_uniform(local_rand_state)),__float2half(curand_uniform(local_rand_state)),__float2half(0)) - vec3_half(__float2half(1),__float2half(1),__float2half(0));
    } while (dot(p,p) >= __float2half(1.0f));
    return p;
}

class camera_half {
public:
    __device__ camera_half(vec3_half lookfrom, vec3_half lookat, vec3_half vup, half vfov, half aspect, half aperture, half focus_dist) { // vfov is top to bottom in degrees
        lens_radius = aperture / __float2half(2.0f);
        half theta = vfov*((half)M_PI)/__float2half(180.0f);
        half half_height = tan(__half2float(theta/__float2half(2.0f)));
        half half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin  - half_width*focus_dist*u -half_height*focus_dist*v - focus_dist*w;
        horizontal = __float2half(2.0f)*half_width*focus_dist*u;
        vertical = __float2half(2.0f)*half_height*focus_dist*v;
    }
    __device__ ray_half get_ray(half s, half t, curandState *local_rand_state) {
        vec3_half rd = lens_radius*random_in_unit_disk_half(local_rand_state);
        vec3_half offset = u * rd.x() + v * rd.y();
        return ray_half(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset);
    }

    vec3_half origin;
    vec3_half lower_left_corner;
    vec3_half horizontal;
    vec3_half vertical;
    vec3_half u, v, w;
    half lens_radius;
};

#endif
