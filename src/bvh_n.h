#ifndef BVH_NH
#define BVH_NH

#ifdef BVH
#include "hitable.h"
#include "aabb.h"

#define BVH_N 10

class bvh_n : public hitable {
public:
    __device__ bvh_n() {}

    __device__ ~bvh_n() {
        if (!_isLeaf) {
            for (int i = 0; i < BVH_N; i++) {
                delete _children[i];
            }
        }
    }
    __device__ bvh_n(hitable** l, int n) : _n_objects(n), _objects(l, n) {
        if (n <= BVH_N) {
            _isLeaf = true;
            _bbox = AABB();
            for (int i = 0; i < n; i++) _bbox.merge_with(l[i]->bbox());
            return;
        }

        AABB aabb = AABB();
        for (int i = 0; i < n; i++)
            aabb.merge_with(l[i]->bbox());

        int axis = aabb.get_longest_axis();

        sort_by_axis(l, n, axis); // It might be more efficient to do it on the CPU

        int c = n / BVH_N;
        _bbox = AABB();
        for (int i = 0; i < BVH_N; i++) {
            _children[i] = new bvh_n(l + i * c, min(c, n - c * i));
            _bbox.merge_with(_children[i]->bbox());
        }
        _isLeaf = false;
    }

    __device__ bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
        if (_isLeaf) {
            return _objects.hit(r, tmin, tmax, rec);
        }
        if (!bbox().hit(r, tmin, tmax))
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
    int _n_objects;
    hitable_list _objects;
    bvh_n* _children[BVH_N];
    bool _isLeaf;
    __device__ void sort_by_axis(hitable** list, int n, int axis) {
        for (int i = 1; i < n; i++) {
            hitable* obj = list[i];
            float obj_center;
            if (axis == 0) obj_center = obj->center().x();
            else if (axis == 1) obj_center = obj->center().y();
            else obj_center = obj->center().z();

            int j = i - 1;
            while (j >= 0) {
                float j_center;
                hitable* j_obj = list[j];
                if (axis == 0) j_center = j_obj->center().x();
                else if (axis == 1) j_center = j_obj->center().y();
                else j_center = j_obj->center().z();

                if (j_center > obj_center) {
                    list[j + 1] = list[j];
                    j--;
                }
                else {
                    break;
                }
            }
            list[j + 1] = obj;
        }
    }
};
#endif
#endif