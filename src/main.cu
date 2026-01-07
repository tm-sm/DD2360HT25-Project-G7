#ifdef REDUCED_PRECISION
#include "aabb_half.h"
#include "bvh_half.h"
#include "camera_half.h"
#include "hitable_half.h"
#include "hitable_list_half.h"
#include "material_half.h"
#include "ray_half.h"
#include "sphere_half.h"
#include "vec3_half.h"

typedef vec3_half vec3;
typedef ray_half ray;
#ifdef BVH
typedef AABB_half AABB;
#endif
typedef hit_record_half hit_record;
typedef hitable_half hitable;
typedef hitable_list_half hitable_list;
typedef material_half material;
typedef lambertian_half lambertian;
typedef metal_half metal;
typedef dielectric_half dielectric;
typedef sphere_half sphere;
typedef camera_half camera;
#ifdef BVH
typedef bvh_half bvh;
#endif
#else
#include "aabb.h"
#include "bvh.h"
#include "bvh_n.h"
#include "camera.h"
#include "hitable.h"
#include "hitable_list.h"
#include "material.h"
#include "ray.h"
#include "sphere.h"
#include "vec3.h"
#endif
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
#ifdef REDUCED_PRECISION
    vec3 cur_attenuation = vec3(__float2half(1.0f), __float2half(1.0f), __float2half(1.0f));
#else
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
#endif
    for (int i = 0; i < BOUNCES; i++) {
        hit_record rec;
#ifdef REDUCED_PRECISION
        if ((*world)->hit(cur_ray, __float2half(0.005f), __float2half(60000.0f), rec)) {
#else
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
#endif
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
#ifdef SMART_BOUNCES
                // Return early if the color is black, a the only return involving cur_attenuation multiplie it by another color
                // and every other return return black already
                if (cur_attenuation.r() < 0.001 && cur_attenuation.g() < 0.001 && cur_attenuation.b() < 0.001)
                    return cur_attenuation;
#endif
                cur_ray = scattered;
            } else {
#ifdef REDUCED_PRECISION
                return vec3(__float2half(0.0f), __float2half(0.0f), __float2half(0.0f));
#else
                return vec3(0.0, 0.0, 0.0);
#endif
            }
        } else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
#ifdef REDUCED_PRECISION
            half t = __float2half(0.5f) * (unit_direction.y() + __float2half(1.0f));
            vec3 c = (__float2half(1.0f) - t) * vec3(__float2half(1.0f), __float2half(1.0f), __float2half(1.0f)) + t * vec3(__float2half(0.5f), __float2half(0.7f), __float2half(1.0f));
#else
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
#endif
            return cur_attenuation * c;
        }
    }
#ifdef REDUCED_PRECISION
    return vec3(__float2half(0.0f), __float2half(0.0f), __float2half(0.0f)); // exceeded recursion
#else
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
#endif
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

#ifndef PARALLEL_RAYS
__global__ void render(vec3 *frame_buffer, int max_x, int max_y, int num_steps, camera **cam, hitable **world, curandState *rand_state) {
    // Determine pixel coordinated (i, j)
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
#ifdef REDUCED_PRECISION
    // Initialize empty color vector for the pixel (FP16)
    vec3 col(__float2half(0.0f), __float2half(0.0f), __float2half(0.0f));
#else
    // Initialize empty color vector for the pixel (FP32)
    vec3 col(0, 0, 0);
#endif
    // Each thread computes all samples for its pixel one by one
    for (int s = 0; s < num_steps; s++) {
#ifdef REDUCED_PRECISION
        // Jitter the ray slightly (offsetting) for anti-aliasing
        half u = __float2half(float(i + curand_uniform(&local_rand_state)) / float(max_x));
        half v = __float2half(float(j + curand_uniform(&local_rand_state)) / float(max_y));
#else
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
#endif
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
#ifdef REDUCED_PRECISION
    // Average color and apply correction (sqrt) - FP16
    col /= __float2half(float(num_steps));
    col[0] = hsqrt(col[0]);
    col[1] = hsqrt(col[1]);
    col[2] = hsqrt(col[2]);
#else
    // Same as above with FP32 
    col /= float(num_steps);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
#endif
    frame_buffer[pixel_index] = col;
}
#else
__global__ void render(vec3 *frame_buffer, int max_x, int max_y, int num_steps, camera **cam, hitable **world, curandState *rand_state) {
    // Determine pixel coordinated (i, j)
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    // Index of sample (ray) within the pixel
    int k = threadIdx.z;
    if ((i >= max_x) || (j >= max_y) || (k >= num_steps))
        return;
    int pixel_index = j * max_x + i;
    int random_index = j * max_x * num_steps + i * num_steps + k;
    curandState local_rand_state = rand_state[random_index];

    // Shared memory (stores every color before summing)
    __shared__ vec3 col_per_ray[THREADS_Y][THREADS_X][RAYS_PER_PIXEL];
#ifdef REDUCED_PRECISION
    // Generate and trace exactly ONE ray per thread (FP16)
    half u = __float2half(float(i + curand_uniform(&local_rand_state)) / float(max_x));
    half v = __float2half(float(j + curand_uniform(&local_rand_state)) / float(max_y));
#else
    // Generate and trace exactly ONE ray per thread (FP32)
    float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
#endif
    ray r = (*cam)->get_ray(u, v, &local_rand_state);
    col_per_ray[threadIdx.y][threadIdx.x][threadIdx.z] = color(r, world, &local_rand_state);
    rand_state[random_index] = local_rand_state;

    if (k != 0)
        return;

    // Ensure every thread has finished its ray cast
    __syncthreads();
#ifdef REDUCED_PRECISION
    vec3 col(__float2half(0.0f), __float2half(0.0f), __float2half(0.0f));
#else
    vec3 col(0, 0, 0);
#endif

    // Compute the final pixel color
    for (int s = 0; s < num_steps; s++) {
        col += col_per_ray[threadIdx.y][threadIdx.x][s];
    }
#ifdef REDUCED_PRECISION
    col /= __float2half(float(num_steps));
    col[0] = hsqrt(col[0]);
    col[1] = hsqrt(col[1]);
    col[2] = hsqrt(col[2]);
#else
    col /= float(num_steps);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
#endif
    frame_buffer[pixel_index] = col;
}
#endif

#ifdef OPTIMIZED_RENDER
__global__ void render_optimized(vec3 *frame_buffer, int max_x, int max_y, int num_steps, camera **cam, hitable **world, curandState *rand_state) {
    // Calaculate unique pixel coordinates for the thread
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    // Boundary Check (ensure threads outside the image dimensions do nothing)
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    // Load random state for this specific pixel into local register memory
    curandState local_rand_state = rand_state[pixel_index];
#ifdef REDUCED_PRECISION
    // Add a small offset to the coordinates to smooth out edges (Anti-Aliasing)
    half u = __float2half(float(i + curand_uniform(&local_rand_state)) / float(max_x));
    half v = __float2half(float(j + curand_uniform(&local_rand_state)) / float(max_y));
#else
    // Same as above with FP32 precision
    float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
#endif
    // Ray and path casting
    ray r = (*cam)->get_ray(u, v, &local_rand_state);
    vec3 col = color(r, world, &local_rand_state);
    rand_state[pixel_index] = local_rand_state;
    // Atomic adding (avoid data races)
    atomicAdd(&frame_buffer[pixel_index].e[0], col.r());
    atomicAdd(&frame_buffer[pixel_index].e[1], col.g());
    atomicAdd(&frame_buffer[pixel_index].e[2], col.b());
}
#endif

#define RND (curand_uniform(&local_rand_state))

// Instantiates the world with geometry and camera objects (using either standard or half precision)
__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
#ifdef REDUCED_PRECISION
        // Creates a sphere at (0, -1000, -1) with radius 1000 (large radius = appears as flat surface)
        d_list[0] = new sphere(vec3(__float2half(0.0f), __float2half(-1000.0f), __float2half(-1.0f)), __float2half(1000.0f),
                               new lambertian(vec3(__float2half(0.5f), __float2half(0.5f), __float2half(0.5f))));
#else
        // Same as above with standard precision
        d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
#endif
        int i = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
#ifdef REDUCED_PRECISION
                // Converting random float to 16-bit half precision float
                half choose_mat = __float2half(RND);
                vec3 center(__float2half(a + RND), __float2half(0.2f), __float2half(b + RND));
                // Defining probabilities of materials for the spheres
                if (choose_mat < __float2half(0.8f)) {
                    // 80% chance for Lambertian (diffuse/matte) material
                    d_list[i++] = new sphere(center, __float2half(0.2f),
                                             new lambertian(vec3(__float2half(RND * RND), __float2half(RND * RND), __float2half(RND * RND))));
                } else if (choose_mat < __float2half(0.95f)) {
                    // 15% chance for metal material
                    d_list[i++] = new sphere(center, __float2half(0.2f),
                                             new metal(vec3(__float2half(0.5f * (1.0f + RND)), __float2half(0.5f * (1.0f + RND)), __float2half(0.5f * (1.0f + RND))), __float2half(0.5f * RND)));
                } else {
                    // 5% chance for glass (dielectric) material
                    d_list[i++] = new sphere(center, __float2half(0.2f), new dielectric(__float2half(1.5f)));
                }
#else
                // Same logic as above with FP32 (full-precision)
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
#endif
            }
        }
#ifdef REDUCED_PRECISION
        // Central glass sphere
        d_list[i++] = new sphere(vec3(__float2half(0.0f), __float2half(1.0f), __float2half(0.0f)), __float2half(1.0f), new dielectric(__float2half(1.5f)));
        // Left matte sphere
        d_list[i++] = new sphere(vec3(__float2half(-4.0f), __float2half(1.0f), __float2half(0.0f)), __float2half(1.0f), new lambertian(vec3(__float2half(0.4f), __float2half(0.2f), __float2half(0.1f))));
        // Right metal sphere
        d_list[i++] = new sphere(vec3(__float2half(4.0f), __float2half(1.0f), __float2half(0.0f)), __float2half(1.0f), new metal(vec3(__float2half(0.7f), __float2half(0.6f), __float2half(0.5f)), __float2half(0.0f)));
#else
        // Same logic as above with FP32
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
#endif
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, 22 * 22 + 1 + 3);
#ifdef REDUCED_PRECISION
        // Physical location of the camera in the 3D world
        vec3 lookfrom(__float2half(13.0f), __float2half(2.0f), __float2half(3.0f));
        // The point the camera is pointed at (center of world)
        vec3 lookat(__float2half(0.0f), __float2half(0.0f), __float2half(0.0f));
        // Distance to the plane where the image is perfectly in focus
        half dist_to_focus = __float2half(10.0f);
        (lookfrom - lookat).length();
        // Size of the lens and camera initialization
        half aperture = __float2half(0.1f);
        *d_camera = new camera(lookfrom,
                               lookat,
                               vec3(__float2half(0.0f), __float2half(1.0f), __float2half(0.0f)),
                               __float2half(30.0f),
                               __float2half(float(nx) / float(ny)),
                               aperture,
                               dist_to_focus);
#else
        // Same as above for FP32 precision
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
#endif
    }
}

// Delete all the scene resources that were created
__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}


#ifdef BVH
// Builds the BVH by initializing nodes and recursively partitioning the scene objects
__global__ void create_bvh(hitable **d_list, int num_obj, bvh_n **d_bvh, int num_bvh) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    // since a setting up a nodea requires their children to be instantiated, we need to create the entire tree and then setup every node
    for (int i = 0; i < num_bvh; i++)
        d_bvh[i] = new bvh_n();
    d_bvh[0]->setup_root(d_list, num_obj);
    for (int i = 0; i < num_bvh; i++)
        d_bvh[i]->setup(i, d_bvh, num_bvh);
}
// Recursively deletes the BVH nodes that were created
__global__ void free_bvh(bvh_n **d_bvh, int num_bvh) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    for (int i = 0; i < num_bvh; i++)
        delete d_bvh[i];
}
#endif

int main(int argc, char const *argv[]) {
    // Change the image size if needed
    int pixels_x = 1200;
    int pixels_y = 800;
    int num_steps = RAYS_PER_PIXEL;
    int tx = THREADS_X;
    int ty = THREADS_Y;

    size_t limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    cudaDeviceSetLimit(cudaLimitStackSize, limit * 2);

    std::cout << "Rendering a " << pixels_x << "x" << pixels_y << " image with " << num_steps << " samples per pixel ";
    std::cout << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = pixels_x * pixels_y;
    size_t fb_size = num_pixels * sizeof(vec3);

    // Allocate Frame Buffer (FB)
    vec3 *d_frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **)&d_frame_buffer, fb_size));
    checkCudaErrors(cudaMemset(d_frame_buffer, 0, fb_size));

    // Allocate Random State
    curandState *d_rand_state;
#ifndef PARALLEL_RAYS
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
#else
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * num_steps * sizeof(curandState)));
#endif
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

    // We need that 2nd random state to be initialized for the world creation
    std::cout << "[1] Initializing the world random generator" << std::endl;
    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "[2] Initializing the world" << std::endl;
    // Make our world of hitables & the camera
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
#ifdef BVH
    // compute the min depth neccessary to store every sphere, knowing the every layer l stores BVH_N^l elements
    int depth = log(num_hitables) / log(BVH_N);
    
    // computes the total number of bvh_n instance needed in our tree
    int num_bvh = 1;
    for (int d = 0; d < depth; d++)
    num_bvh = num_bvh * BVH_N + 1;
    
    // allocates the memory for the bvh_n tree
    bvh_n **d_bvh;
    checkCudaErrors(cudaMalloc((void **)&d_bvh, num_bvh * sizeof(bvh_n *)));
    create_bvh<<<1, 1>>>(d_list, num_hitables, d_bvh, num_bvh);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
#endif

    clock_t start, stop;
    start = clock();
    // Render our buffer

    std::cout << "[3] Initializing the pixels random generator" << std::endl;
#ifndef PARALLEL_RAYS
    dim3 blocks(pixels_x / tx + 1, pixels_y / ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(pixels_x, pixels_y, d_rand_state);
#else
    // use a 3d grid of tx * ty * num_steps instead of a 2d grid
    dim3 blocks(pixels_x / tx + 1, pixels_y / ty + 1, 1);
    dim3 threads(tx, ty, num_steps);
    render_init<<<blocks, threads>>>(pixels_x, pixels_y, num_steps, d_rand_state);
#endif
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "[4] Rendering the frame" << std::endl;
#ifdef OPTIMIZED_RENDER
    // Call render 5 times and do a single pass each times
    for (int s = 0; s < num_steps; s++) {
        render_optimized<<<blocks, threads>>>(d_frame_buffer, pixels_x, pixels_y, num_steps, d_camera, d_world, d_rand_state);
    }
#elifdef PARALLEL_RAYS
    // Specify the correct amount of shared memory: sizeof(vec3) bytes per pixel per step
    render<<<blocks, threads, tx * ty * num_steps * sizeof(vec3)>>>(d_frame_buffer, pixels_x, pixels_y, num_steps, d_camera, d_world, d_rand_state);
#elifdef BVH
    // Render using BVH tree (instead of a raw list)t
    render<<<blocks, threads>>>(d_frame_buffer, pixels_x, pixels_y, num_steps, d_camera, (hitable **)d_bvh, d_rand_state);
#else
    // Standard rendering
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
        std::cout << "[5] Creating file \"" << argv[1] << "\" \n";
        file.open(argv[1]);

        // PPM Header: ASCII color (P3) - Width & Height - Max Color Value 
        file << "P3\n"
             << pixels_x << " " << pixels_y << "\n255\n";
        for (int j = pixels_y - 1; j >= 0; j--) {
            for (int i = 0; i < pixels_x; i++) {
                size_t pixel_index = j * pixels_x + i;
                vec3 col = d_frame_buffer[pixel_index];

#ifdef OPTIMIZED_RENDER
#ifdef REDUCED_PRECISION
// Average the color by the number of samples and apply 
orrection ([h]sqrt) (to make shadows look cleaner)                col /= __float2half(float(num_steps));
                col = vec3(hsqrt(col[0]), hsqrt(col[1]), hsqrt(col[2]));
#else
                col /= float(num_steps);
                col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
#endif
#endif
#ifdef REDUCED_PRECISION
                int ir =
// Colors need to be precise and thus converting back to float int(255.99 * __half2float(col.r()));
                int ig = int(255.99 * __half2float(col.g()));
                int ib = int(255.99 * __half2float(col.b()));
#else
                int ir = int(255.99 * col.r());
                int ig = int(255.99 * col.g());
                int ib = int(255.99 * col.b());
#endif
                file << ir << " " << ig << " " << ib << "\n";
            }
        }

        file.close();
    }

    // Cleanup: Call free_world kernel to delete objects created    std::cout << "[6] Cleaning up" << std::endl;
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
#ifdef BVH
    free_bvh<<<1, 1>>>(d_bvh, num_bvh);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
#endif
    // Free all allocated device memory
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(d_frame_buffer));

    cudaDeviceReset();
}
