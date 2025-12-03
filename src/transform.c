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
    // Start with identity
    glm_mat4_identity(out);

    // Apply translation
    glm_translate(out, t->translation);

    // Apply rotations (XYZ order)
    glm_rotate_x(out, t->rotation[0]);
    glm_rotate_y(out, t->rotation[1]);
    glm_rotate_z(out, t->rotation[2]);

    // Apply scale
    glm_scale(out, t->scale);
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
    vec3 right, down, normal;

    glm_vec3_sub(planePts[1], planePts[0], right);   // TR - TL
    glm_vec3_sub(planePts[3], planePts[0], down);    // BL - TL

    float sx = (w > 1) ? 1.0f / (w - 1) : 1.0f;
    float sz = (h > 1) ? 1.0f / (h - 1) : 1.0f;

    glm_vec3_scale(right, sx, right);
    glm_vec3_scale(down,  sz, down);

    glm_mat4_identity(t->matrix);

    t->matrix[0][0] = right[0];
    t->matrix[1][0] = right[1];
    t->matrix[2][0] = right[2];

    glm_vec3_cross(right, down, normal);
    glm_vec3_normalize(normal);

    t->matrix[0][1] = normal[0];
    t->matrix[1][1] = normal[1];
    t->matrix[2][1] = normal[2];

    t->matrix[0][2] = down[0];
    t->matrix[1][2] = down[1];
    t->matrix[2][2] = down[2];

    t->matrix[3][0] = planePts[0][0];
    t->matrix[3][1] = planePts[0][1];
    t->matrix[3][2] = planePts[0][2];
}
