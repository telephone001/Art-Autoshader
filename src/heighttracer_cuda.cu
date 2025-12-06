// heighttracer_cuda.cu
#include "heighttracer_cuda.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h> 

// ------------------ Constant memory ------------------
__constant__ Camera d_cam_const;
__constant__ float HMAP_DX_SCALE = 1.0f;

// ------------------ Helpers ------------------
__device__ inline float sample_hmap_nearest(const float* hmap, int w, int l, float x, float z)
{
    int ix = (int)floorf(x);
    int iz = (int)floorf(z);
    if (ix < 0 || iz < 0 || ix >= w || iz >= l) return -CUDART_INF_F;
    return hmap[iz * w + ix];
}

__device__ inline void vec3s_normalize_inplace(vec3s* v)
{
    float len = sqrtf(v->x * v->x + v->y * v->y + v->z * v->z);
    if (len > 1e-9f) { v->x /= len; v->y /= len; v->z /= len; }
}

// ------------------ Triangle-based normal ------------------
__device__ inline vec3s get_hmap_normal_device_triangles(const float* hmap, int w, int l, float px, float pz)
{
    int ix = (int)floorf(px);
    int iz = (int)floorf(pz);

    if (ix < 0 || iz < 0 || ix >= w - 1 || iz >= l - 1)
        return { 0.0f,1.0f,0.0f };

    float h00 = sample_hmap_nearest(hmap, w, l, (float)ix, (float)iz);
    float h10 = sample_hmap_nearest(hmap, w, l, (float)(ix + 1), (float)iz);
    float h01 = sample_hmap_nearest(hmap, w, l, (float)ix, (float)(iz + 1));
    float h11 = sample_hmap_nearest(hmap, w, l, (float)(ix + 1), (float)(iz + 1));

    float local_x = px - ix;
    float local_z = pz - iz;

    vec3s v0, v1, v2;

    if (local_x + local_z < 1.0f) {
        v0 = { 0.0f,h00,0.0f };
        v1 = { 1.0f,h10,0.0f };
        v2 = { 0.0f,h01,1.0f };
    }
    else {
        v0 = { 1.0f,h11,1.0f };
        v1 = { 0.0f,h01,1.0f };
        v2 = { 1.0f,h10,0.0f };
    }

    vec3s e1 = { v1.x - v0.x, v1.y - v0.y, v1.z - v0.z };
    vec3s e2 = { v2.x - v0.x, v2.y - v0.y, v2.z - v0.z };

    vec3s n = { e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x };
    vec3s_normalize_inplace(&n);
    return n;
}

// ------------------ Ray intersection ------------------
__device__ inline int intersect_heightmap_ray_device(
    const float* hmap, int hm_w, int hm_l,
    vec3s origin, vec3s dir,
    float step, float max_t,
    float* out_t, vec3s* out_p)
{
    float t = 0.0f;
    while (t < max_t)
    {
        float px = origin.x + dir.x * t;
        float py = origin.y + dir.y * t;
        float pz = origin.z + dir.z * t;

        float h = sample_hmap_nearest(hmap, hm_w, hm_l, px, pz);
        // FIX: Use CUDART_INF_F
        if (h == -CUDART_INF_F) return 0;

        if (py <= h)
        {
            if (out_t) *out_t = t;
            if (out_p) { out_p->x = px; out_p->y = py; out_p->z = pz; }
            return 1;
        }
        t += step;
    }
    return 0;
}

// ------------------ Kernel ------------------
__global__ void heightmap_tracer_kernel(
    const float* hmap, int hm_w, int hm_l,
    int screenW, int screenH,
    float step, float max_t,
    float* out_t,
    vec3s* out_hit_points,
    vec3s* out_normals)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = screenW * screenH;
    if (idx >= total) return;

    int py = idx / screenW;
    int px = idx % screenW;

    float radFov = FOVY * (CUDART_PI_F / 180.0f);
    float tanFov = tanf(radFov * 0.5f);
    float aspect = (float)screenW / (float)screenH;

    float ndc_x = ((px + 0.5f) / (float)screenW) * 2.0f - 1.0f;
    float ndc_y = 1.0f - ((py + 0.5f) / (float)screenH) * 2.0f;

    float cam_x = ndc_x * aspect * tanFov;
    float cam_y = ndc_y * tanFov;

    vec3s dir = {
        d_cam_const.front.x + d_cam_const.right.x * cam_x + d_cam_const.up.x * cam_y,
        d_cam_const.front.y + d_cam_const.right.y * cam_x + d_cam_const.up.y * cam_y,
        d_cam_const.front.z + d_cam_const.right.z * cam_x + d_cam_const.up.z * cam_y
    };
    vec3s_normalize_inplace(&dir);

    float t_hit = 0.0f;
    vec3s hitp = { 0.0f,0.0f,0.0f };
    int hit = intersect_heightmap_ray_device(hmap, hm_w, hm_l, d_cam_const.pos, dir, step, max_t, &t_hit, &hitp);

    if (!hit)
    {
        out_t[idx] = -1.0f;
        out_hit_points[idx] = { 0.0f,0.0f,0.0f };
        out_normals[idx] = { 0.0f,1.0f,0.0f };
        return;
    }

    // Binary refinement
    const int REFINEMENT_ITERS = 6;
    float t_lo = (t_hit > step) ? t_hit - step : 0.0f;
    float t_hi = t_hit;
    for (int i = 0; i < REFINEMENT_ITERS; i++)
    {
        float t_mid = 0.5f * (t_lo + t_hi);
        float pxm = d_cam_const.pos.x + dir.x * t_mid;
        float pym = d_cam_const.pos.y + dir.y * t_mid;
        float pzm = d_cam_const.pos.z + dir.z * t_mid;
        float hm = sample_hmap_nearest(hmap, hm_w, hm_l, pxm, pzm);
        if (hm != -CUDART_INF_F && pym <= hm) t_hi = t_mid;
        else t_lo = t_mid;
    }

    float t_final = t_hi;
    vec3s pf = { d_cam_const.pos.x + dir.x * t_final,
                 d_cam_const.pos.y + dir.y * t_final,
                 d_cam_const.pos.z + dir.z * t_final };

    vec3s normal = get_hmap_normal_device_triangles(hmap, hm_w, hm_l, pf.x, pf.z);

    out_t[idx] = t_final;
    out_hit_points[idx] = pf;
    out_normals[idx] = normal;
}

// ------------------ Host wrapper (FIXED) ------------------
void ht_trace_all_cuda(
    const float* hmap_host, int hm_w, int hm_l,
    const Camera* cam_host,
    int screenW, int screenH,
    float step, float max_t,
    float** out_t_ptr_host,
    vec3s** out_hit_points_ptr_host,
    vec3s** out_normals_ptr_host)
{
    // --- 1. DECLARATIONS MUST BE FIRST (Before any goto) ---
    // Even primitive types like int must be declared here to satisfy E0546

    int total_pixels = screenW * screenH;
    size_t t_size = (size_t)total_pixels * sizeof(float);
    size_t p_size = (size_t)total_pixels * sizeof(vec3s);
    size_t hmap_size = (size_t)hm_w * (size_t)hm_l * sizeof(float);

    // Device pointers
    float* d_hmap = NULL;
    float* d_out_t = NULL;
    vec3s* d_hit = NULL;
    vec3s* d_norm = NULL;

    // Host pointers
    float* out_t_host = NULL;
    vec3s* out_hit_host = NULL;
    vec3s* out_norm_host = NULL;
    float* initbuf = NULL;

    // Temporary struct for copying
    Camera cam_tmp;

    // Kernel configuration
    int THREADS = 256;
    int BLOCKS = (total_pixels + THREADS - 1) / THREADS;

    // --- 2. LOGIC START ---

    // Check input validity
    if (!hmap_host || !cam_host || screenW <= 0 || screenH <= 0) return;

    // --- Allocate Device Memory ---
    if (cudaMalloc(&d_hmap, hmap_size) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_out_t, t_size) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_hit, p_size) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_norm, p_size) != cudaSuccess) goto cleanup;

    // --- Copy Data to Device ---
    if (cudaMemcpy(d_hmap, hmap_host, hmap_size, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;

    // Copy Camera constant
    // NOTE: E0413 warning here is a common IntelliSense false positive in CUDA. 
    // The code IS valid and will compile.
    cam_tmp = *cam_host;
    if (cudaMemcpyToSymbol(d_cam_const, &cam_tmp, sizeof(Camera)) != cudaSuccess) goto cleanup;

    // --- Initialize Device Output Buffers ---
    initbuf = (float*)malloc(t_size);
    if (!initbuf) goto cleanup;

    // Initialize array contents on host before copying
    for (int i = 0; i < total_pixels; i++) initbuf[i] = -1.0f;
    if (cudaMemcpy(d_out_t, initbuf, t_size, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;

    free(initbuf); initbuf = NULL; // Free temporary buffer

    if (cudaMemset(d_hit, 0, p_size) != cudaSuccess) goto cleanup;
    if (cudaMemset(d_norm, 0, p_size) != cudaSuccess) goto cleanup;

    // --- Launch Kernel ---
    heightmap_tracer_kernel << <BLOCKS, THREADS >> > (d_hmap, hm_w, hm_l, screenW, screenH, step, max_t, d_out_t, d_hit, d_norm);

    // Synchronize to catch async errors
    if (cudaDeviceSynchronize() != cudaSuccess) goto cleanup;

    // --- Allocate Host Output Memory ---
    out_t_host = (float*)malloc(t_size);
    out_hit_host = (vec3s*)malloc(p_size);
    out_norm_host = (vec3s*)malloc(p_size);
    if (!out_t_host || !out_hit_host || !out_norm_host) goto cleanup;

    // --- Copy Device Results to Host ---
    if (cudaMemcpy(out_t_host, d_out_t, t_size, cudaMemcpyDeviceToHost) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(out_hit_host, d_hit, p_size, cudaMemcpyDeviceToHost) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(out_norm_host, d_norm, p_size, cudaMemcpyDeviceToHost) != cudaSuccess) goto cleanup;

    // --- Assign Pointers to Caller (Success) ---
    if (out_t_ptr_host) *out_t_ptr_host = out_t_host;
    if (out_hit_points_ptr_host) *out_hit_points_ptr_host = out_hit_host;
    if (out_normals_ptr_host) *out_normals_ptr_host = out_norm_host;

    // Set allocated host pointers to NULL so they are skipped in cleanup (only device cleanup remains)
    out_t_host = NULL;
    out_hit_host = NULL;
    out_norm_host = NULL;

cleanup:
    // --- Centralized Cleanup ---

    // Free all host memory if it failed *after* allocation
    if (initbuf) free(initbuf);
    if (out_t_host) free(out_t_host);
    if (out_hit_host) free(out_hit_host);
    if (out_norm_host) free(out_norm_host);

    // Free all device memory
    if (d_hmap) cudaFree(d_hmap);
    if (d_out_t) cudaFree(d_out_t);
    if (d_hit) cudaFree(d_hit);
    if (d_norm) cudaFree(d_norm);

    // On any failure, ensure external pointers are NULL
    if (cudaPeekAtLastError() != cudaSuccess && cudaPeekAtLastError() != cudaErrorCudartUnloading)
    {
        if (out_t_ptr_host) *out_t_ptr_host = NULL;
        if (out_hit_points_ptr_host) *out_hit_points_ptr_host = NULL;
        if (out_normals_ptr_host) *out_normals_ptr_host = NULL;
    }
}