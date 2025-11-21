#include "transform.h"
#include <cglm/cglm.h>
#include <string.h>

// Basic TRS transform
void hmap_transform_compute(HeightmapTransform* t)
{
    glm_mat4_identity(t->matrix);

    // Scale
    glm_scale(t->matrix, t->scale);

    // Rotate (XYZ order)
    glm_rotate_x(t->matrix, t->rotation[0], t->matrix);
    glm_rotate_y(t->matrix, t->rotation[1], t->matrix);
    glm_rotate_z(t->matrix, t->rotation[2], t->matrix);

    // Translation
    glm_translate(t->matrix, t->translation);
}

// map the heightmap to a 3d plane using the 4 corners
// planePts = { TL, TR, BR, BL }
// heightmap grid spans:
//   x: 0..hmap_width
//   z: 0..hmap_height
// This computes a model matrix that linearly interpolates the plane

void hmap_transform_from_plane(
    HeightmapTransform* out,
    vec3 planePts[4],
    int w, int h
) {
    // 1. Compute basis vectors of the plane
    vec3 right; // TR - TL
    vec3 down;  // BL - TL
    glm_vec3_sub(planePts[1], planePts[0], right);
    glm_vec3_sub(planePts[3], planePts[0], down);

    // Scale basis to heightmap size
    glm_vec3_scale(right, 1.0f / (float)w, right);
    glm_vec3_scale(down,  1.0f / (float)h, down);

    // 2. Build model matrix using TL as origin
    glm_mat4_identity(out->matrix);

    // Insert basis vectors as matrix columns
    out->matrix[0][0] = right[0];
    out->matrix[1][0] = right[1];
    out->matrix[2][0] = right[2];

    out->matrix[0][2] = down[0];
    out->matrix[1][2] = down[1];
    out->matrix[2][2] = down[2];

    // Up vector (normal)
    vec3 normal;
    glm_vec3_cross(right, down, normal);
    glm_vec3_normalize(normal);

    out->matrix[0][1] = normal[0];
    out->matrix[1][1] = normal[1];
    out->matrix[2][1] = normal[2];

    // 3. Set translation = TL of plane
    out->matrix[3][0] = planePts[0][0];
    out->matrix[3][1] = planePts[0][1];
    out->matrix[3][2] = planePts[0][2];
}
