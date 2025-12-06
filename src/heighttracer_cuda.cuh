#pragma once
#include "glfw_window.h" // Includes Camera and vec3s (from cglm)

#ifdef __cplusplus
extern "C" {
#endif

void ht_trace_all_cuda(
    const float* hmap_host, int hm_w, int hm_l,
    const Camera* cam_host,
    int screenW, int screenH,
    float step, float max_t,
    float** out_t_ptr_host, 
    vec3s** out_points_ptr_host
);

#ifdef __cplusplus
}
#endif