#include "heighttracer_cuda.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h> 

// CRITICAL FIX: Use constant memory for read-only Camera struct
__constant__ Camera d_cam_const;

// ---------------- IntelliSense Compatibility ----------------
#ifdef __INTELLISENSE__
#define __global__
#define __device__
#define __host__
#define __shared__
#define CUDA_KERNEL_LAUNCH(kernel, grid, block, ...) 
#else
#define CUDA_KERNEL_LAUNCH(kernel, grid, block, ...) \
    kernel<<<grid, block>>>(__VA_ARGS__)
#endif
// ------------------------------------------------------------


// ---------------- Device helpers ----------------
__device__ float sample_hmap_nearest(
    const float* hmap, int w, int l, float x, float z)
{
    // Check if coordinates are within the bounds [0, w-1] and [0, l-1]
    int ix = (int)floorf(x);
    int iz = (int)floorf(z);
    
    if (ix < 0 || iz < 0 || ix >= w || iz >= l) return -INFINITY;
    
    return hmap[iz * w + ix];
}

__device__ void vec3s_normalize(vec3s* v)
{
    float len = sqrtf(v->x*v->x + v->y*v->y + v->z*v->z);
    if (len > 1e-6f) {
        v->x /= len;
        v->y /= len;
        v->z /= len;
    }
}

__device__ int intersect_heightmap_ray_device(
    const float* hmap, int hm_w, int hm_l,
    vec3s origin, vec3s dir,
    float step, float max_t,
    float* out_t, vec3s* out_p)
{
    // Note: CPU version starts raymarching from t=0.0f, but the CUDA
    // version previously started at t=step. Reverting to t=0.0f 
    // to perfectly match the CPU logic.
    float t = 0.0f; 

    while (t < max_t) {
        float px = origin.x + dir.x * t;
        float py = origin.y + dir.y * t;
        float pz = origin.z + dir.z * t;

        float h = sample_hmap_nearest(hmap, hm_w, hm_l, px, pz);

        if (h == -INFINITY) {
            return 0; 
        }

        // if py <= surface height -> hit
        if (py <= h) {
            *out_t = t;
            out_p->x = px; out_p->y = py; out_p->z = pz;
            return 1;
        }

        t += step;
    }
    return 0;
}
// ------------------------------------------------------------


// ------------------- Kernel ----------------------
__global__ void heightmap_tracer_kernel(
    const float* hmap, int hm_w, int hm_l,
    int screenW, int screenH,
    float step, float max_t,
    float* out_t,
    vec3s* out_points)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = screenW * screenH;
    if (idx >= total_pixels) return;

    int py = idx / screenW;
    int px = idx % screenW;
    
    // FOVY is from glfw_window.h
    float radFov = FOVY * (CUDART_PI_F / 180.0f);
    float tanFov = tanf(radFov * 0.5f);
    float aspect = (float)screenW / (float)screenH;

    // Map pixel coordinates to Normalized Device Coordinates (NDC)
    float ndc_x = ((px + 0.5f) / screenW) * 2.f - 1.f;
    float ndc_y = 1.f - ((py + 0.5f) / screenH) * 2.f; 
    
    // Convert NDC to View Space coordinates (Z=-1 plane)
    float cam_x = ndc_x * aspect * tanFov;
    float cam_y = ndc_y * tanFov;

    vec3s dir;
    // CRITICAL FIX: Match CPU logic for World Space ray direction.
    // The CPU version implicitly uses 1.0f for the forward component,
    // which fixes the direction reversal bug.
    dir.x = d_cam_const.front.x * 1.0f + d_cam_const.right.x * cam_x + d_cam_const.up.x * cam_y;
    dir.y = d_cam_const.front.y * 1.0f + d_cam_const.right.y * cam_x + d_cam_const.up.y * cam_y;
    dir.z = d_cam_const.front.z * 1.0f + d_cam_const.right.z * cam_x + d_cam_const.up.z * cam_y;

    vec3s_normalize(&dir);

    float t_hit;
    vec3s hitp;

    // Raymarch to find initial hit (t_hit)
    int hit = intersect_heightmap_ray_device(
        hmap, hm_w, hm_l,
        d_cam_const.pos, dir, 
        step, max_t,
        &t_hit, &hitp
    );

    if (!hit) {
        out_t[idx] = -1.f;
        out_points[idx].x = 0.f;
        out_points[idx].y = 0.f;
        out_points[idx].z = 0.f;
        return;
    }

    // Binary search refinement
    const int REFINEMENT_ITERS = 6;
    // t_lo must start from 0 if t_hit is the first step hit.
    float t_lo = t_hit > step ? t_hit - step : 0.0f;
    float t_hi = t_hit;

    for (int i = 0; i < REFINEMENT_ITERS; i++) {
        float t_mid = 0.5f * (t_lo + t_hi);

        float pxm = d_cam_const.pos.x + dir.x * t_mid;
        float pym = d_cam_const.pos.y + dir.y * t_mid;
        float pzm = d_cam_const.pos.z + dir.z * t_mid;

        float hm = sample_hmap_nearest(hmap, hm_w, hm_l, pxm, pzm);

        if (hm != -INFINITY && pym <= hm)
            t_hi = t_mid;
        else
            t_lo = t_mid;
    }

    float t_final = t_hi;

    vec3s pf;
    pf.x = d_cam_const.pos.x + dir.x * t_final;
    pf.y = d_cam_const.pos.y + dir.y * t_final;
    pf.z = d_cam_const.pos.z + dir.z * t_final;

    out_t[idx] = t_final;
    out_points[idx] = pf;
}
// ------------------------------------------------------------


// ---------------- Host Wrapper --------------------
// (No change here from previous successful version, retaining error checks and __constant__ memcpy)
void ht_trace_all_cuda(
    const float* hmap_host, int hm_w, int hm_l,
    const Camera* cam_host,
    int screenW, int screenH,
    float step, float max_t,
    float** out_t_ptr_host,
    vec3s** out_points_ptr_host)
{
    // ---- declare EVERYTHING up front ----
    int total_pixels;
    int THREADS;
    int BLOCKS;
    size_t t_size;
    size_t p_size;
    size_t hmap_size;
    float* hmap_device = NULL;
    float* out_t_device = NULL;
    vec3s* out_points_device = NULL;
    float* initbuf = NULL;
    cudaError_t err;
    
    *out_t_ptr_host = NULL;
    *out_points_ptr_host = NULL;

    if (!hmap_host || screenW <= 0 || screenH <= 0 || !cam_host) return;

    total_pixels = screenW * screenH;
    t_size = (size_t)total_pixels * sizeof(float);
    p_size = (size_t)total_pixels * sizeof(vec3s);
    hmap_size = (size_t)hm_w * (size_t)hm_l * sizeof(float);

    // --- Allocate GPU memory ---
    err = cudaMalloc((void**)&hmap_device, hmap_size);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA Malloc hmap_device failed: %s\n", cudaGetErrorString(err)); goto fail; }

    err = cudaMalloc((void**)&out_t_device, t_size);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA Malloc out_t_device failed: %s\n", cudaGetErrorString(err)); goto fail; }

    err = cudaMalloc((void**)&out_points_device, p_size);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA Malloc out_points_device failed: %s\n", cudaGetErrorString(err)); goto fail; }

    // --- Upload heightmap and Camera data ---
    err = cudaMemcpy(hmap_device, hmap_host, hmap_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA Memcpy hmap_host failed: %s\n", cudaGetErrorString(err)); goto fail; }

    // CRITICAL FIX: Upload Camera data to __constant__ memory
    err = cudaMemcpyToSymbol(d_cam_const, cam_host, sizeof(Camera));
    if (err != cudaSuccess) { fprintf(stderr, "CUDA MemcpyToSymbol d_cam_const failed: %s\n", cudaGetErrorString(err)); goto fail; }

    // Init -1 buffer for out_t_device (to guarantee initial miss status)
    initbuf = (float*)malloc(t_size);
    if (!initbuf) { fprintf(stderr, "Host Malloc initbuf failed.\n"); goto fail; }
    for (int i = 0; i < total_pixels; i++) initbuf[i] = -1.f;

    err = cudaMemcpy(out_t_device, initbuf, t_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA Memcpy initbuf failed: %s\n", cudaGetErrorString(err)); goto fail; }

    free(initbuf);
    initbuf = NULL;

    cudaMemset(out_points_device, 0, p_size);

    // ---- Kernel launch ----
    THREADS = 256;
    BLOCKS = (total_pixels + THREADS - 1) / THREADS;

    CUDA_KERNEL_LAUNCH(heightmap_tracer_kernel,
        BLOCKS, THREADS,
        hmap_device, hm_w, hm_l,
        screenW, screenH,
        step, max_t,
        out_t_device,
        out_points_device
    );

#ifndef __INTELLISENSE__
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Execution Error: %s\n", cudaGetErrorString(err));
        goto fail;
    }
#endif

    // ---- Copy back ----
    float* out_t_host = (float*)malloc(t_size);
    vec3s* out_points_host = (vec3s*)malloc(p_size);
    if (!out_t_host || !out_points_host) {
        fprintf(stderr, "Host Malloc final buffers failed.\n");
        free(out_t_host); 
        free(out_points_host);
        goto fail;
    }

    err = cudaMemcpy(out_t_host, out_t_device, t_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { 
        fprintf(stderr, "CUDA Memcpy back out_t_host failed: %s\n", cudaGetErrorString(err));
        free(out_t_host); 
        free(out_points_host);
        goto fail;
    }

    err = cudaMemcpy(out_points_host, out_points_device, p_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { 
        fprintf(stderr, "CUDA Memcpy back out_points_host failed: %s\n", cudaGetErrorString(err));
        free(out_t_host); 
        free(out_points_host);
        goto fail;
    }

    *out_t_ptr_host = out_t_host;
    *out_points_ptr_host = out_points_host;

    // Successful path cleanup
    cudaFree(hmap_device);
    cudaFree(out_t_device);
    cudaFree(out_points_device);
    return;

fail:
    // Error path cleanup
    if (initbuf) free(initbuf);
    if (hmap_device) cudaFree(hmap_device);
    if (out_t_device) cudaFree(out_t_device);
    if (out_points_device) cudaFree(out_points_device);
    
    // Set pointers to NULL to signal failure to the caller
    *out_t_ptr_host = NULL;
    *out_points_ptr_host = NULL;
    return;
}