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
    // Compute two axes along the plane from the first corner
    vec3 uVec, vVec;
    glm_vec3_sub(planePts[1], planePts[0], uVec); // vector toward second corner
    glm_vec3_sub(planePts[3], planePts[0], vVec); // vector toward fourth corner

    float uLen = glm_vec3_norm(uVec);
    float vLen = glm_vec3_norm(vVec);

    if (uLen < 1e-6f || vLen < 1e-6f)
        return;

    // Normalize directions
    vec3 uDir, vDir;
    glm_vec3_normalize_to(uVec, uDir);
    glm_vec3_normalize_to(vVec, vDir);

    // Step size for each heightmap cell
    vec3 uStep, vStep;
    glm_vec3_scale(uDir, uLen / (w - 1), uStep);
    glm_vec3_scale(vDir, vLen / (h - 1), vStep);

    // Compute normal (height axis)
    vec3 normal;
    glm_vec3_cross(uDir, vDir, normal);
    glm_vec3_normalize(normal);

    float heightScale = 0.075f * t->scale[1];
    vec3 heightAxis;
    glm_vec3_scale(normal, heightScale, heightAxis);

    // Fill transform matrix
    glm_mat4_identity(t->matrix);

    // Width axis
    t->matrix[0][0] = uStep[0];
    t->matrix[1][0] = uStep[1];
    t->matrix[2][0] = uStep[2];

    // Height axis
    t->matrix[0][1] = heightAxis[0];
    t->matrix[1][1] = heightAxis[1];
    t->matrix[2][1] = heightAxis[2];

    // Depth axis
    t->matrix[0][2] = vStep[0];
    t->matrix[1][2] = vStep[1];
    t->matrix[2][2] = vStep[2];

    // Translation (origin corner)
    glm_vec3_copy(planePts[0], t->matrix[3]);
}
