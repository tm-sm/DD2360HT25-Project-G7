#include "camera.h"
#include "hitable_list.h"
#include "bvh.h"
#include "bvh_n.h"
#include "aabb.h"
#include "material.h"
#include "ray.h"
#include "sphere.h"
#include "vec3.h"
#include <curand_kernel.h>
#include <float.h>
#include <fstream>
#include <iostream>
#include <time.h>

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

#ifndef THREADS_X
#define THREADS_X 8
#endif

#ifndef THREADS_Y
#define THREADS_Y 8
#endif

#ifndef RAYS_PER_PIXEL
#define RAYS_PER_PIXEL 10
#endif

#ifndef BOUNCES
#define BOUNCES 50
#endif

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray &r, hitable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    // TODO 10 seams to be a sweet spot
    for (int i = 0; i < BOUNCES; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return vec3(0.0, 0.0, 0.0);
            }
        } else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

#ifndef PARALLEL_RAYS
__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
#else
__global__ void render_init(int max_x, int max_y, int max_s, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z;
    if ((i >= max_x) || (j >= max_y) || (k >= max_s))
        return;
    int pixel_index = j * max_x * max_s + i * max_s + k;

#endif
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

// ? New alg for better performances

// TODO try split the loop into multiple threads and sum after @MIDHU
#ifndef PARALLEL_RAYS
__global__ void render(vec3 *frame_buffer, int max_x, int max_y, int num_steps, camera **cam, hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < num_steps; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(num_steps);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    frame_buffer[pixel_index] = col;
}
#else
__global__ void render(vec3 *frame_buffer, int max_x, int max_y, int num_steps, camera **cam, hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z;
    if ((i >= max_x) || (j >= max_y) || (k >= num_steps))
        return;
    int pixel_index = j * max_x + i;
    int random_index = j * max_x * num_steps + i * num_steps + k;
    curandState local_rand_state = rand_state[random_index];

    __shared__ vec3 col_per_ray[THREADS_Y][THREADS_X][RAYS_PER_PIXEL];
    float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
    ray r = (*cam)->get_ray(u, v, &local_rand_state);
    col_per_ray[threadIdx.y][threadIdx.x][threadIdx.z] = color(r, world, &local_rand_state);
    rand_state[random_index] = local_rand_state;

    if (k != 0)
        return;
    __syncthreads();
    vec3 col(0, 0, 0);
    for (int s = 0; s < num_steps; s++) {
        col += col_per_ray[threadIdx.y][threadIdx.x][s];
    }
    col /= float(num_steps);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    frame_buffer[pixel_index] = col;
}
#endif

// optimized render function - RESULT: NO IMPROVEMENT! @MIDHU
__global__ void render_optimized(vec3 *frame_buffer, int max_x, int max_y, int num_steps, camera **cam, hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
    ray r = (*cam)->get_ray(u, v, &local_rand_state);
    vec3 col = color(r, world, &local_rand_state);
    rand_state[pixel_index] = local_rand_state;
    atomicAdd(&frame_buffer[pixel_index].e[0], col.r());
    atomicAdd(&frame_buffer[pixel_index].e[1], col.g());
    atomicAdd(&frame_buffer[pixel_index].e[2], col.b());
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
                } else if (choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                } else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
#ifdef BVH
        *d_world = new bvh(d_list, 22 * 22 + 1 + 3);
#else
        *d_world = new hitable_list(d_list, 22 * 22 + 1 + 3);
#endif
        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0;
        (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(lookfrom,
                               lookat,
                               vec3(0, 1, 0),
                               30.0,
                               float(nx) / float(ny),
                               aperture,
                               dist_to_focus);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main(int argc, char const *argv[]) {
    int pixels_x = 1200;
    int pixels_y = 800;
    int num_steps = RAYS_PER_PIXEL;
    int tx = THREADS_X;
    int ty = THREADS_Y;

    std::cout << "Rendering a " << pixels_x << "x" << pixels_y << " image with " << num_steps << " samples per pixel ";
    std::cout << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = pixels_x * pixels_y;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate FB
    vec3 *d_frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **)&d_frame_buffer, fb_size));
    checkCudaErrors(cudaMemset(d_frame_buffer, 0, fb_size));

    // allocate random state
    curandState *d_rand_state;
#ifndef PARALLEL_RAYS
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
#else
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * num_steps * sizeof(curandState)));
#endif
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    std::cout << "[1] Initializing the world random generator" << std::endl;
    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the camera
    hitable **d_list;
    int num_hitables = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables * sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera, pixels_x, pixels_y, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer

    // TODO can be improved maybe possible eventually @MIDHU
    std::cout << "[2] Initializing the pixels random generator" << std::endl;
#ifndef PARALLEL_RAYS
    dim3 blocks(pixels_x / tx + 1, pixels_y / ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(pixels_x, pixels_y, d_rand_state);
#else
    dim3 blocks(pixels_x / tx + 1, pixels_y / ty + 1, 1);
    dim3 threads(tx, ty, num_steps);
    render_init<<<blocks, threads>>>(pixels_x, pixels_y, num_steps, d_rand_state);
#endif
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "[3] Rendering the frame" << std::endl;
#ifdef USE_OPTIMIZED_RENDER
    for (int s = 0; s < num_steps; s++) {
        render_optimized<<<blocks, threads>>>(d_frame_buffer, pixels_x, pixels_y, num_steps, d_camera, d_world, d_rand_state);
    }
#elifdef PARALLEL_RAYS
    render<<<blocks, threads, tx * ty * num_steps * sizeof(vec3)>>>(d_frame_buffer, pixels_x, pixels_y, num_steps, d_camera, d_world, d_rand_state);
#else
    render<<<blocks, threads>>>(d_frame_buffer, pixels_x, pixels_y, num_steps, d_camera, d_world, d_rand_state);
#endif
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cout << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    std::ofstream file;
    if (argc == 2) {
        std::cout << "Creating file \"" << argv[1] << "\" \n";
        file.open(argv[1]);

        file << "P3\n"
             << pixels_x << " " << pixels_y << "\n255\n";
        for (int j = pixels_y - 1; j >= 0; j--) {
            for (int i = 0; i < pixels_x; i++) {
                size_t pixel_index = j * pixels_x + i;
                vec3 col = d_frame_buffer[pixel_index];
#ifdef USE_OPTIMIZED_RENDER
                col /= float(num_steps);
                col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
#endif
                int ir = int(255.99 * col.r());
                int ig = int(255.99 * col.g());
                int ib = int(255.99 * col.b());
                file << ir << " " << ig << " " << ib << "\n";
            }
        }

        file.close();
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(d_frame_buffer));

    cudaDeviceReset();
}
