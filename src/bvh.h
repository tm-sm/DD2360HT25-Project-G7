#ifndef BVHH
#define BVHH

#ifdef BVH
#include "hitable.h"
#include "aabb.h"

class bvh : public hitable {
public:
    __device__ bvh() {}
    __device__ bvh(hitable* object) {}

    __device__ ~bvh() {
        if (!isLeaf) {
            delete _left;
            delete _right;
        }
    }
    __device__ bvh(hitable** l, int n) {
        if (n == 1) {
            isLeaf = true;
            object = l[0];
            _bbox = AABB(l[0]->bbox());
            return;
        }

        AABB aabb = AABB(l[0]->bbox());
        for (int i = 1; i < n; i++)
            aabb.merge_with(l[i]->bbox());

        int axis = aabb.get_longest_axis();

        sort_by_axis(l, n, axis); // It might be more efficient to do it on the CPU

        // n = 3 n/2=1
        int m = n / 2;
        _left = new bvh(l, m); // Not sure if we can give this to different cores/threads to parallelize
        _right = new bvh(l + m, n - m);

        isLeaf = false;
        _bbox = AABB(_left->bbox(), _right->bbox());
    }

    __device__ bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
        if (isLeaf) {
            return object->hit(r, tmin, tmax, rec);
        }
        const bvh* current = this;
        while (!current->isLeaf) {
            if (current->_left->bbox().hit(r)) current = current->_left;
            else if (current->_right->bbox().hit(r)) current = current->_right;
            else return false;
        }
        return current->object->hit(r, tmin, tmax, rec);
    }
    hitable* object;
    bool isLeaf;
    bvh* _left;
    bvh* _right;
protected:
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

    __device__ bool hit_node(const ray& r, float t_min, float t_max, hit_record& rec) const {
        if (!bbox().hit(r))
            return false;

        if (isLeaf)
            return object->hit(r, t_min, t_max, rec);

        hit_record left_rec, right_rec;
        bool hit_left = _left->hit_node(r, t_min, t_max, left_rec);
        bool hit_right = _right->hit_node(r, t_min, t_max, right_rec);

        if (hit_left && hit_right) {
            rec = (left_rec.t < right_rec.t) ? left_rec : right_rec;
            return true;
        }
        else if (hit_left) {
            rec = left_rec;
            return true;
        }
        else if (hit_right) {
            rec = right_rec;
            return true;
        }

        return false;
    }

};
#endif
#endif