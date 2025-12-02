#ifndef BVHH
#define BVHH

#include "hitable.h"

class bvh_node {
public:
    vec3 center;
    float radius;
    bvh_node* left;
    bvh_node* right;
};

class bvh : public hitable {
public:
    __device__ bvh() {}
    __device__ bvh(hitable** l, int n) {} // TODO: use a bottom up method to create the tree
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    // TODO: Maybe add a second class that can do left-right paths
    bvh_node* root;
};

__device__ bool bvh::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    // TODO: implement the actual collision check
    return false;
}

#endif