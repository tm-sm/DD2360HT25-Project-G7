#ifndef HITABLELIST_HALFH
#define HITABLELIST_HALFH

#include "hitable_half.h"

class hitable_list_half: public hitable_half  {
    public:
        __device__ hitable_list_half() {}
        __device__ hitable_list_half(hitable_half **l, int n) {list = l; list_size = n; }
        __device__ virtual bool hit(const ray_half& r, half tmin, half tmax, hit_record_half& rec) const;
        hitable_half **list;
        int list_size;
};


__device__ bool hitable_list_half::hit(const ray_half& r, half t_min, half t_max, hit_record_half& rec) const {
        hit_record_half temp_rec;
        bool hit_anything = false;
        half closest_so_far = t_max;
        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
}

#endif
