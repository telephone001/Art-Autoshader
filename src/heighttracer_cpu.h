#pragma once

#include "editor.h"     // Editor, editor_get_* accessors
#include "glfw_window.h"// Camera type
#include <stdlib.h>     // malloc/free

// NOTE: uses vec3s from cglm (type used across your project)

// Allocates and returns an array of ray directions (vec3s) for every pixel.
// Length = screenW * screenH.
// Caller MUST free() the returned pointer when done.
vec3s* ht_generate_camera_directions(const Camera* cam, int screenW, int screenH);

// Intersect a single ray (origin + direction) against a heightmap.
// Returns 1 if hit, 0 if miss. On hit out_t and out_p are filled.
int ht_intersect_heightmap_ray(
    const float* hmap, int hm_w, int hm_l,
    const vec3s origin, const vec3s dir,
    float step, float max_t,
    float* out_t, vec3s* out_p
);

// Trace all camera rays (screenW * screenH) against heightmap.
// Allocates out_t array (float) and out_points array (vec3s). Caller must free.
// out_t length = screenW * screenH (value -1.0f for MISS), out_points length = screenW * screenH.
void ht_trace_all(
    const float* hmap, int hm_w, int hm_l,
    const Camera* cam,
    int screenW, int screenH,
    float step, float max_t,
    float** out_t_ptr,    // returned pointer-to-array (malloc'd)
    vec3s** out_points_ptr// returned pointer-to-array (malloc'd)
);
