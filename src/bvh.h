#ifndef BVHH
#define BVHH

#include "hitable.h"
#include "aabb.h"

class bvh_node {
public:
    AABB* aabb; //Implement an actual AABB
    hitable* node;
    bool isLeaf;
    bvh_node* left;
    bvh_node* right;
};

class bvh : public hitable {
public:
    __device__ bvh() {}
    __device__ bvh_node* build_recursive(hitable** l, int n) {
        bvh_node* node = new bvh_node();
        if (n == 1) {
            node->isLeaf = true;
            node->node = l[0];
            node->aabb = AABB(l, 1);
            return node;
        }

        AABB* aabb = AABB(l, n);
        /* TODO: impement these two functions

        int axis = get_longest_axis(aabb);

        sort_by_axis(l, n, axis); // It might be more efficient to do it on the CPU

         */

        int m = n/2;
        bvh_node* left = build_recursive(l, m); // Not sure if we can give this to different cores/threads to parallelize
        bvh_node* right = build_recursive(l + m, n - m);

        node->isLeaf = false;
        node->left = left;
        node->right = right;
        node->aabb = AABB(left->aabb, right->aabb);

        return node;
    }

    __device__ bvh(hitable** l, int n) {
        root = build_recursive(l, n);
    }
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    bvh_node* root;
};

__device__ bool bvh::hit_node(bvh_node* node, const ray& r, float t_min, float t_max, hit_record& rec) const {
    if (!node) return false;

    if (!node->aabb.hit(r, t_min, t_max))
        return false;

    if (node->isLeaf) {
        return node->node->hit(r, t_min, t_max, rec);
    }

    hit_record left_rec, right_rec;
    bool hit_left = hit_node(node->left, r, t_min, t_max, left_rec);
    bool hit_right = hit_node(node->right, r, t_min, t_max, right_rec);

    if (hit_left && hit_right) {
        rec = (left_rec.t < right_rec.t) ? left_rec : right_rec;
        return true;
    } else if (hit_left) {
        rec = left_rec;
        return true;
    } else if (hit_right) {
        rec = right_rec;
        return true;
    }

    return false;
}

__device__ bool bvh::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    return hit_node(root, r, t_min, t_max, rec);
}

#endif