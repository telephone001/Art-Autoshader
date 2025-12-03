#include "transform.h"
#include <cglm/cglm.h>

void transform_init(Transform* t) {
    glm_vec3_zero(t->translation);
    glm_vec3_zero(t->rotation);
    t->scale[0] = t->scale[1] = t->scale[2] = 1.0f;
    glm_mat4_identity(t->matrix);
}

void transform_get_matrix(const Transform* t, mat4 out)
{
    glm_mat4_identity(out);

    // Translation requires non-const vec3
    vec3 trans = { t->translation[0], t->translation[1], t->translation[2] };
    glm_translate(out, trans);

    // Rotations require dest matrix argument
    glm_rotate_x(out, t->rotation[0], out);
    glm_rotate_y(out, t->rotation[1], out);
    glm_rotate_z(out, t->rotation[2], out);

    // Scale also requires non-const vec3
    vec3 scl = { t->scale[0], t->scale[1], t->scale[2] };
    glm_scale(out, scl);
}

void hmap_transform_compute(HeightmapTransform* t) {
    glm_mat4_identity(t->matrix);

    glm_translate(t->matrix, t->translation);
    glm_rotate_x(t->matrix, t->rotation[0], t->matrix);
    glm_rotate_y(t->matrix, t->rotation[1], t->matrix);
    glm_rotate_z(t->matrix, t->rotation[2], t->matrix);
    glm_scale(t->matrix, t->scale);
}

void hmap_transform_from_plane(
    HeightmapTransform* t,
    vec3 planePts[4],     // {TL, TR, BR, BL}
    int w,                // width  (columns)
    int h                 // height (rows)
) {
    // --- Compute plane axes ---
    vec3 rightVec, downVec;

    // TR - TL  → plane width direction
    glm_vec3_sub(planePts[1], planePts[0], rightVec);
    // BL - TL  → plane height direction
    glm_vec3_sub(planePts[3], planePts[0], downVec);

    // --- Scale factors: one step in heightmap space ---
    float sx = (w > 1) ? 1.0f / (float)(w - 1) : 1.0f;
    float sz = (h > 1) ? 1.0f / (float)(h - 1) : 1.0f;

    // --- Scale axes so that 0→w-1 maps to the whole plane ---
    vec3 rightStep, downStep;
    glm_vec3_scale(rightVec, sx, rightStep);
    glm_vec3_scale(downVec,  sz, downStep);

    // --- Height direction: use normal scaled by height_scale ---
    vec3 normal;
    glm_vec3_cross(rightVec, downVec, normal);
    glm_vec3_normalize(normal);
    glm_vec3_scale(normal, t->height_scale, normal);

    // --- Build matrix ---
    glm_mat4_identity(t->matrix);

    // Column 0 → X direction (right)
    t->matrix[0][0] = rightStep[0];
    t->matrix[1][0] = rightStep[1];
    t->matrix[2][0] = rightStep[2];

    // Column 1 → Height direction (normal scaled)
    t->matrix[0][1] = normal[0];
    t->matrix[1][1] = normal[1];
    t->matrix[2][1] = normal[2];

    // Column 2 → Z direction (down)
    t->matrix[0][2] = downStep[0];
    t->matrix[1][2] = downStep[1];
    t->matrix[2][2] = downStep[2];

    // Column 3 → translation (TL point)
    t->matrix[3][0] = planePts[0][0];
    t->matrix[3][1] = planePts[0][1];
    t->matrix[3][2] = planePts[0][2];
}
