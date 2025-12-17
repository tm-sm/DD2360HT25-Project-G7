#ifndef BVH_NH
#define BVH_NH

#ifdef BVH
#include "hitable_list.h"
#include "aabb.h"

#ifndef BVH_N
#define BVH_N 8
#endif

class bvh_n : public hitable {
public:
    __device__ void setup_root(hitable** l, int n) {
        _objects = hitable_list(l, n);
    }

    __device__ void setup(int id, bvh_n** tree, int tree_n) {
        // Assumes the parent has already been setup
        _bbox = AABB();
        for (int i = 0; i < _objects.list_size; i++)
            _bbox.merge_with(_objects.list[i]->bbox());

        _isLeaf = id * BVH_N+1 >= tree_n;
        if (_isLeaf) return;

        _children = tree + (id * BVH_N + 1);
        sort_by_axis(_objects.list, _objects.list_size, _bbox.get_longest_axis()); // It might be more efficient to do it on the CPU

        int c = _objects.list_size / BVH_N;
        int rem = _objects.list_size % BVH_N;
        int start = 0;
        for (int i = 0; i < BVH_N; i++) {
            int count = i < rem ? (c + 1) : c;
            _children[i]->_objects = hitable_list(_objects.list + start, count);
            start += count;
        }
    }

    __device__ bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
        if (_isLeaf) {
            return _objects.hit(r, tmin, tmax, rec);
        }

        if (!bbox().hit(r))
            return false;

        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = tmax;
        for (int i = 0; i < BVH_N; i++) {
            if (_children[i]->hit(r, tmin, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }
protected:
    hitable_list _objects;
    bvh_n** _children;
    bool _isLeaf;
    __device__ void sort_by_axis(hitable** list, int n, int axis) {
        for (int i = 0; i < n - 1; i++) {
            float min_center = list[i]->center().e[axis];
            int min = i;
            for (int j = i + 1; j < n; j++) {
                float j_center = list[j]->center().e[axis];
                if (j_center < min_center) {
                    min = j;
                    min_center = j_center;
                }
            }
            hitable* tmp = list[i];
            list[i] = list[min];
            list[min] = tmp;
        }
    }
};
#endif
#endif