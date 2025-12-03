#pragma once
#include <cglm/cglm.h>

// Alias so existing project code can use "Transform"
typedef struct HeightmapTransform {
    vec3 translation;   // final translation in world space
    vec3 rotation;      // euler rotations (radians)
    vec3 scale;         // scale in x,y,z
    mat4 matrix;        // final model matrix
    float x_axis[3];
    float y_axis[3];
    float z_axis[3];
    float origin[3];
    float height_scale;
    float matrix[4][4];
} HeightmapTransform;

typedef HeightmapTransform Transform;

// Initialize transform to identity
void transform_init(Transform* t);

// Computes TRS into t->matrix
void transform_get_matrix(const Transform* t, mat4 out);

// Fills transform->matrix with transform computed from TRS
void hmap_transform_compute(HeightmapTransform* transform);

// Computes a placement transform that maps the heightmap grid 
// onto the 4 plane points (top-left, top-right, bottom-right, bottom-left)
void hmap_transform_from_plane(
    HeightmapTransform* transform,
    vec3 planePts[4],
    int hmap_width,
    int hmap_height
);
