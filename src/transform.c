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
    vec3 corners[4],     // {TL, TR, BR, BL}
    int w,               // heightmap width  (columns)
    int h                // heightmap height (rows)
)
{
    vec3 xDir, yDir, normal;

    // 1. Compute plane directions (TR - TL, BL - TL)
    glm_vec3_sub(corners[1], corners[0], xDir);  // X direction across top edge
    glm_vec3_sub(corners[3], corners[0], yDir);  // Y direction down left edge

    // Normalize both
    glm_vec3_normalize(xDir);
    glm_vec3_normalize(yDir);

    // 2. Compute normal (right × down)
    glm_vec3_cross(xDir, yDir, normal);
    glm_vec3_normalize(normal);

    // 3. Scale axes to match heightmap grid spacing
    float sx = (w > 1) ? 1.0f / (w - 1) : 1.0f;
    float sy = (h > 1) ? 1.0f / (h - 1) : 1.0f;

    glm_vec3_scale(xDir, sx, t->x_axis);
    glm_vec3_scale(yDir, sy, t->y_axis);
    glm_vec3_scale(normal, t->height_scale, t->z_axis);
    // 4. Set origin exactly at TL corner
    glm_vec3_copy(corners[0], t->origin);
    // 5. Build 4×4 matrix
    glm_mat4_identity(t->matrix);

    // X column
    t->matrix[0][0] = t->x_axis[0];
    t->matrix[1][0] = t->x_axis[1];
    t->matrix[2][0] = t->x_axis[2];

    // Y column
    t->matrix[0][1] = t->y_axis[0];
    t->matrix[1][1] = t->y_axis[1];
    t->matrix[2][1] = t->y_axis[2];

    // Z column (height)
    t->matrix[0][2] = t->z_axis[0];
    t->matrix[1][2] = t->z_axis[1];
    t->matrix[2][2] = t->z_axis[2];

    // Origin / translation
    t->matrix[3][0] = t->origin[0];
    t->matrix[3][1] = t->origin[1];
    t->matrix[3][2] = t->origin[2];
}
