#include "transform.h"
#include <cglm/cglm.h>

/* ---------------------------------------------------------
   Initialize transform to identity
--------------------------------------------------------- */
void transform_init(Transform* t) {
    glm_vec3_zero(t->translation);
    glm_vec3_zero(t->rotation);
    t->scale[0] = t->scale[1] = t->scale[2] = 1.0f;
    glm_mat4_identity(t->matrix);

    // Optional axis parameters (not required for rendering)
    glm_vec3_zero(t->x_axis);
    glm_vec3_zero(t->y_axis);
    glm_vec3_zero(t->z_axis);
    glm_vec3_zero(t->origin);

    t->height_scale = 1.0f;
}

/* ---------------------------------------------------------
   Build TRS matrix into 'out'
--------------------------------------------------------- */
void transform_get_matrix(const Transform* t, mat4 out)
{
    mat4 T, R, S;

    // Translation
    glm_translate_make(T, t->translation);

    // Euler rotation
    glm_euler_xyz(t->rotation, R);

    // Scale
    glm_scale_make(S, t->scale);

    // out = T * R * S
    glm_mat4_mulN((mat4 *[]){ T, R, S }, 3, out);
}

/* ---------------------------------------------------------
   Update t->matrix using TRS stored inside transform
--------------------------------------------------------- */
void hmap_transform_compute(HeightmapTransform* t)
{
    transform_get_matrix(t, t->matrix);
}

/* ---------------------------------------------------------
   Helper: Transform 4 vec3 points by a matrix
--------------------------------------------------------- */
static void transform_plane_points(mat4 model, vec3 inPts[4], vec3 outPts[4])
{
    for (int i = 0; i < 4; i++) {
        vec4 p = { inPts[i][0], inPts[i][1], inPts[i][2], 1.0f };
        vec4 r;

        glm_mat4_mulv(model, p, r);

        outPts[i][0] = r[0];
        outPts[i][1] = r[1];
        outPts[i][2] = r[2];
    }
}

/* ---------------------------------------------------------
   IMPORTANT: Your original heightmap placement transform
   (I KEEP THIS EXACTLY AS YOU WROTE IT)
--------------------------------------------------------- */
void hmap_transform_from_plane(
    HeightmapTransform* t,
    vec3 planePts[4],
    int w,
    int h
) {
    // Compute two vectors along the plane edges from the origin corner
    vec3 uVec, vVec;
    glm_vec3_sub(planePts[1], planePts[0], uVec); // width direction
    glm_vec3_sub(planePts[3], planePts[0], vVec); // height direction

    // Step size for each heightmap cell
    vec3 uStep, vStep;
    glm_vec3_scale(uVec, 1.0f / (w - 1), uStep);
    glm_vec3_scale(vVec, 1.0f / (h - 1), vStep);

    // Compute normal (height axis)
    vec3 normal;
    glm_vec3_cross(uVec, vVec, normal);
    glm_vec3_normalize(normal);

    float heightScale = 0.075f * t->scale[1];
    vec3 heightAxis;
    glm_vec3_scale(normal, heightScale, heightAxis);

    // Fill matrix
    glm_mat4_identity(t->matrix);

    // U-axis
    t->matrix[0][0] = uStep[0];
    t->matrix[1][0] = uStep[1];
    t->matrix[2][0] = uStep[2];

    // Height axis
    t->matrix[0][1] = heightAxis[0];
    t->matrix[1][1] = heightAxis[1];
    t->matrix[2][1] = heightAxis[2];

    // V-axis
    t->matrix[0][2] = vStep[0];
    t->matrix[1][2] = vStep[1];
    t->matrix[2][2] = vStep[2];

    // Translation (origin)
    glm_vec3_copy(planePts[0], t->matrix[3]);
}
