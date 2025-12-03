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
    vec3 planePts[4],     // {TL, TR, BR, BL}
    int w,                // heightmap width samples
    int h                 // heightmap length samples
) {
    vec3 rightVec, downVec;

    glm_vec3_sub(planePts[1], planePts[0], rightVec); // TR - TL
    glm_vec3_sub(planePts[3], planePts[0], downVec);  // BL - TL

    float planeW = glm_vec3_norm(rightVec);
    float planeH = glm_vec3_norm(downVec);

    if (planeW < 1e-6f || planeH < 1e-6f)
        return;

    glm_vec3_normalize(rightVec);
    glm_vec3_normalize(downVec);

    // world-space step size for each cell
    vec3 rightStep, downStep;
    glm_vec3_scale(rightVec, planeW / (w - 1), rightStep);
    glm_vec3_scale(downVec,  planeH / (h - 1), downStep);

    vec3 normal;
    glm_vec3_cross(downVec, rightVec, normal);
    glm_vec3_normalize(normal);

    float heightScale = 0.075f * t->scale[1];
    vec3 heightAxis;
    glm_vec3_scale(normal, heightScale, heightAxis);

    glm_mat4_identity(t->matrix);

    // X axis = right steps
    t->matrix[0][0] = rightStep[0];
    t->matrix[1][0] = rightStep[1];
    t->matrix[2][0] = rightStep[2];

    // Y axis = height axis
    t->matrix[0][1] = heightAxis[0];
    t->matrix[1][1] = heightAxis[1];
    t->matrix[2][1] = heightAxis[2];

    // Z axis = down steps
    t->matrix[0][2] = downStep[0];
    t->matrix[1][2] = downStep[1];
    t->matrix[2][2] = downStep[2];

    // translation
    t->matrix[3][0] = planePts[0][0];
    t->matrix[3][1] = planePts[0][1];
    t->matrix[3][2] = planePts[0][2];
}

