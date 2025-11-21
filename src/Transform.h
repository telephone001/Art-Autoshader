#pragma once
#include <cglm/cglm.h>

typedef struct HeightmapTransform {
    vec3 translation;   // final translation in world space
    vec3 rotation;      // euler rotations (radians)
    vec3 scale;         // scale in x,y,z
    mat4 matrix;        // final model matrix
} HeightmapTransform;

// Fills transform->matrix with transform computed from TRS
void hmap_transform_compute(HeightmapTransform* transform);

// Computes a placement transform that maps the heightmap grid onto the 4 plane points (top-left, top-right, bottom-right, bottom-left)
void hmap_transform_from_plane(
    HeightmapTransform* transform,
    vec3 planePts[4],
    int hmap_width,
    int hmap_height
);
