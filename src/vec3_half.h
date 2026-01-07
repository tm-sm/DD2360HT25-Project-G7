#ifndef VEC3_HALFH
#define VEC3_HALFH

#include <cuda_fp16.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>

class vec3_half  {


public:
    __host__ __device__ vec3_half() {}
    __host__ __device__ vec3_half(half e0, half e1, half e2) {
        e[0] = e0; e[1] = e1; e[2] = e2;
#ifdef LENGTH_CACHING
        e[3] = 0;
#endif
    }
    __host__ __device__ inline void set_x(half x) { e[0] = x; }
    __host__ __device__ inline void set_y(half y) { e[1] = y; }
    __host__ __device__ inline void set_z(half z) { e[2] = z; }
    __host__ __device__ inline void set_r(half r) { e[0] = r; }
    __host__ __device__ inline void set_g(half g) { e[1] = g; }
    __host__ __device__ inline void set_b(half b) { e[2] = b; }
    __host__ __device__ inline half x() const { return e[0]; }
    __host__ __device__ inline half y() const { return e[1]; }
    __host__ __device__ inline half z() const { return e[2]; }
    __host__ __device__ inline half r() const { return e[0]; }
    __host__ __device__ inline half g() const { return e[1]; }
    __host__ __device__ inline half b() const { return e[2]; }

    __host__ __device__ inline const vec3_half& operator+() const { return *this; }
    __host__ __device__ inline vec3_half operator-() const { return vec3_half(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline half operator[](int i) const { return e[i]; }
    __host__ __device__ inline half& operator[](int i) { return e[i]; };

    __host__ __device__ inline vec3_half& operator+=(const vec3_half &v2);
    __host__ __device__ inline vec3_half& operator-=(const vec3_half &v2);
    __host__ __device__ inline vec3_half& operator*=(const vec3_half &v2);
    __host__ __device__ inline vec3_half& operator/=(const vec3_half &v2);
    __host__ __device__ inline vec3_half& operator*=(const half t);
    __host__ __device__ inline vec3_half& operator/=(const half t);

    __host__ __device__ inline half length() const;
    __host__ __device__ inline half squared_length() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
    __host__ __device__ inline void make_unit_vector();

#ifndef LENGTH_CACHING
    half e[3];
#else
    half e[4];
#endif
};

__host__ __device__ inline half vec3_half::length() const {
#if defined(__CUDA_ARCH__)
    return hsqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
#else
    return __float2half(sqrtf(__half2float(e[0])*__half2float(e[0]) + __half2float(e[1])*__half2float(e[1]) + __half2float(e[2])*__half2float(e[2])));
#endif
}

inline std::istream& operator>>(std::istream &is, vec3_half &t) {
    float e0, e1, e2;
    is >> e0 >> e1 >> e2;
    t.e[0] = __float2half(e0);
    t.e[1] = __float2half(e1);
    t.e[2] = __float2half(e2);
    return is;
}

inline std::ostream& operator<<(std::ostream &os, const vec3_half &t) {
    os << __half2float(t.e[0]) << " " << __half2float(t.e[1]) << " " << __half2float(t.e[2]);
    return os;
}

__host__ __device__ inline void vec3_half::make_unit_vector() {
    half k = __float2half(1.0f) / length();
    e[0] *= k; e[1] *= k; e[2] *= k;
}

__host__ __device__ inline vec3_half operator+(const vec3_half &v1, const vec3_half &v2) {
    return vec3_half(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3_half operator-(const vec3_half &v1, const vec3_half &v2) {
    return vec3_half(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3_half operator*(const vec3_half &v1, const vec3_half &v2) {
    return vec3_half(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3_half operator/(const vec3_half &v1, const vec3_half &v2) {
    return vec3_half(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3_half operator*(half t, const vec3_half &v) {
    return vec3_half(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3_half operator/(vec3_half v, half t) {
    return vec3_half(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

__host__ __device__ inline vec3_half operator*(const vec3_half &v, half t) {
    return vec3_half(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline half dot(const vec3_half &v1, const vec3_half &v2) {
    return v1.e[0] *v2.e[0] + v1.e[1] *v2.e[1]  + v1.e[2] *v2.e[2];
}

__host__ __device__ inline vec3_half cross(const vec3_half &v1, const vec3_half &v2) {
    return vec3_half( (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
                (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])),
                (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));
}


__host__ __device__ inline vec3_half& vec3_half::operator+=(const vec3_half &v){
    e[0]  += v.e[0];
    e[1]  += v.e[1];
    e[2]  += v.e[2];
    return *this;
}

__host__ __device__ inline vec3_half& vec3_half::operator*=(const vec3_half &v){
    e[0]  *= v.e[0];
    e[1]  *= v.e[1];
    e[2]  *= v.e[2];
    return *this;
}

__host__ __device__ inline vec3_half& vec3_half::operator/=(const vec3_half &v){
    e[0]  /= v.e[0];
    e[1]  /= v.e[1];
    e[2]  /= v.e[2];
    return *this;
}

__host__ __device__ inline vec3_half& vec3_half::operator-=(const vec3_half& v) {
    e[0]  -= v.e[0];
    e[1]  -= v.e[1];
    e[2]  -= v.e[2];
    return *this;
}

__host__ __device__ inline vec3_half& vec3_half::operator*=(const half t) {
    e[0]  *= t;
    e[1]  *= t;
    e[2]  *= t;
    return *this;
}

__host__ __device__ inline vec3_half& vec3_half::operator/=(const half t) {
    half k = __float2half(1.0f)/t;

    e[0]  *= k;
    e[1]  *= k;
    e[2]  *= k;
    return *this;
}

__host__ __device__ inline vec3_half unit_vector(vec3_half v) {
    return v / v.length();
}

__host__ __device__ inline vec3_half reflect(const vec3_half& v, const vec3_half& n) {
    return v - __float2half(2.0f) * dot(v, n) * n;
}

#endif