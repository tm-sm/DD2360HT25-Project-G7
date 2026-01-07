#ifndef MATERIAL_HALFH
#define MATERIAL_HALFH

#include <curand_kernel.h>
#include "ray_half.h"
#include "hitable_half.h"
#include "vec3_half.h"


__device__ half schlick_half(half cosine, half ref_idx) {
    half r0 = (__float2half(1.0f) - ref_idx) / (__float2half(1.0f) + ref_idx);
    r0 = r0 * r0;
    half term = __float2half(1.0f) - cosine;
    return r0 + (__float2half(1.0f) - r0) * term * term * term * term * term;
}

__device__ bool refract_half(const vec3_half& v, const vec3_half& n, half ni_over_nt, vec3_half& refracted) {
    vec3_half uv = unit_vector(v);
    half dt = dot(uv, n);
    half discriminant = __float2half(1.0f) - ni_over_nt * ni_over_nt * (__float2half(1.0f) - dt * dt);
    if (__half2float(discriminant) > 0) {
#if defined(__CUDA_ARCH__)
        refracted = ni_over_nt * (uv - n * dt) - n * hsqrt(discriminant);
#else
        refracted = ni_over_nt * (uv - n * dt) - n * __float2half(sqrtf(__half2float(discriminant)));
#endif
        return true;
    }
    else
        return false;
}

__device__ vec3_half random_in_unit_sphere_half(curandState* local_rand_state) {
    vec3_half p;
    do {
        p = __float2half(2.0f) * vec3_half(__float2half(curand_uniform(local_rand_state)),__float2half(curand_uniform(local_rand_state)),__float2half(curand_uniform(local_rand_state))) - vec3_half(__float2half(1.0f), __float2half(1.0f), __float2half(1.0f));
    } while (p.squared_length() >= __float2half(1.0f));
    return p;
}

class material_half {
public:
    __device__ virtual bool scatter(const ray_half& r_in, const hit_record_half& rec, vec3_half& attenuation, ray_half& scattered, curandState* local_rand_state) const = 0;
};

class lambertian_half : public material_half {
public:
    __device__ lambertian_half(const vec3_half& a) : albedo(a) {}
    __device__ virtual bool scatter(const ray_half& r_in, const hit_record_half& rec, vec3_half& attenuation, ray_half& scattered, curandState* local_rand_state) const {
        vec3_half target = rec.p + rec.normal + random_in_unit_sphere_half(local_rand_state);
        scattered = ray_half(rec.p, target - rec.p);
        attenuation = albedo;
        return true;
    }

    vec3_half albedo;
};

class metal_half : public material_half {
public:
    __device__ metal_half(const vec3_half& a, half f) : albedo(a) { if (__half2float(f) < 1.0f) fuzz = f; else fuzz = __float2half(1.0f); }
    __device__ virtual bool scatter(const ray_half& r_in, const hit_record_half& rec, vec3_half& attenuation, ray_half& scattered, curandState* local_rand_state) const {
        vec3_half reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray_half(rec.p, reflected + fuzz * random_in_unit_sphere_half(local_rand_state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > __float2half(0.0f));
    }
    vec3_half albedo;
    half fuzz;
};

class dielectric_half : public material_half {
public:
    __device__ dielectric_half(half ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(const ray_half& r_in,
        const hit_record_half& rec,
        vec3_half& attenuation,
        ray_half& scattered,
        curandState* local_rand_state) const {
        vec3_half outward_normal;
        vec3_half reflected = reflect(r_in.direction(), rec.normal);
        half ni_over_nt;
        attenuation = vec3_half(__float2half(1.0f), __float2half(1.0f), __float2half(1.0f));
        vec3_half refracted;
        half reflect_prob;
        half cosine;
        if (dot(r_in.direction(), rec.normal) > __float2half(0.0f)) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
#if defined(__CUDA_ARCH__)
            cosine = hsqrt(__float2half(1.0f) - ref_idx * ref_idx * (__float2half(1.0f) - cosine * cosine));
#else
            cosine = __float2half(sqrtf(__half2float(__float2half(1.0f) - ref_idx * ref_idx * (__float2half(1.0f) - cosine * cosine))));
#endif
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = __float2half(1.0f) / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        if (refract_half(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick_half(cosine, ref_idx);
        else
            reflect_prob = __float2half(1.0f);
        if (__float2half(curand_uniform(local_rand_state)) < reflect_prob)
            scattered = ray_half(rec.p, reflected);
        else
            scattered = ray_half(rec.p, refracted);
        return true;
    }

    half ref_idx;
};
#endif