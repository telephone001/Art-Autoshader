#include "transform.h"
#include <cglm/cglm.h>

void transform_init(Transform* t) {
    glm_vec3_zero(t->translation);
    glm_vec3_zero(t->rotation);
    t->scale[0] = t->scale[1] = t->scale[2] = 1.0f;
    glm_mat4_identity(t->matrix);
}

void hmap_transform_from_plane(
    HeightmapTransform* t,
    vec3 planePts[4],     // {corner0, corner1, corner2, corner3}
    int w,                // heightmap width samples
    int h                 // heightmap height samples
) {
    // Compute two vectors along the plane edges from the origin corner
    vec3 uVec, vVec;
    glm_vec3_sub(planePts[1], planePts[0], uVec); // vector toward "width" corner
    glm_vec3_sub(planePts[3], planePts[0], vVec); // vector toward "height" corner

    // Step size for each heightmap cell (keep orientation intact)
    vec3 uStep, vStep;
    glm_vec3_scale(uVec, 1.0f / (w - 1), uStep);
    glm_vec3_scale(vVec, 1.0f / (h - 1), vStep);

    // Compute normal (height axis) using cross product of edges
    vec3 normal;
    glm_vec3_cross(uVec, vVec, normal);
    glm_vec3_normalize(normal);  // just the direction

    float heightScale = 0.075f * t->scale[1];
    vec3 heightAxis;
    glm_vec3_scale(normal, heightScale, heightAxis);

    // Fill transform matrix
    glm_mat4_identity(t->matrix);

    // Width axis (u)
    t->matrix[0][0] = uStep[0];
    t->matrix[1][0] = uStep[1];
    t->matrix[2][0] = uStep[2];

    // Height axis (heightAxis)
    t->matrix[0][1] = heightAxis[0];
    t->matrix[1][1] = heightAxis[1];
    t->matrix[2][1] = heightAxis[2];

    // Depth axis (v)
    t->matrix[0][2] = vStep[0];
    t->matrix[1][2] = vStep[1];
    t->matrix[2][2] = vStep[2];

    // Translation (origin)
    glm_vec3_copy(planePts[0], t->matrix[3]);
}
