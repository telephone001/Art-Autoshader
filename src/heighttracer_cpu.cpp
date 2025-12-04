#include "heighttracer_cpu.h"


// Helper: sample nearest neighbor from heightmap (x,z world coords)
static inline float sample_hmap_nearest(const float* hmap, int w, int l, float x, float z)
{
    int ix = (int)floorf(x);
    int iz = (int)floorf(z);

    if (ix < 0 || iz < 0 || ix >= w || iz >= l) return -INFINITY;
    return hmap[iz * w + ix];
}

// Normalize vec3s (returns length)
static inline float vec3s_len(vec3s* v)
{
    return sqrtf(v->x*v->x + v->y*v->y + v->z*v->z);
}

static inline void vec3s_normalize_inplace(vec3s* v)
{
    float len = vec3s_len(v);
    if (len > 1e-9f) {
        v->x /= len; v->y /= len; v->z /= len;
    }
}

// ------------------------------------------------------------
// Generate directions for each pixel in NDC -> camera plane.
// Uses camera's front/right/up basis, vertical FOV (FOVY, degrees).
// ------------------------------------------------------------
vec3s* ht_generate_camera_directions(const Camera* cam, int screenW, int screenH)
{
    int total = screenW * screenH;
    vec3s* dirs = (vec3s*) malloc(sizeof(vec3s) * total);
    if (!dirs) return NULL;

    // precompute tan(fov/2)
    // FOVY in degrees is a project-wide macro; convert to radians
    float tanFov = tanf(glm_rad(FOVY) * 0.5f); // using glm_rad macro available in project

    float aspect = (float)screenW / (float)screenH;

    // camera basis vectors
    vec3s forward = cam->front;
    vec3s right   = cam->right;
    vec3s up      = cam->up;

    // For every pixel compute NDC and direction in world space
    for (int py = 0; py < screenH; ++py) {
        for (int px = 0; px < screenW; ++px) {
            int idx = py * screenW + px;

            // NDC in [-1,1]
            float ndc_x = ((px + 0.5f) / (float)screenW) * 2.0f - 1.0f;
            float ndc_y = 1.0f - ((py + 0.5f) / (float)screenH) * 2.0f;

            // camera space direction
            float cam_x = ndc_x * aspect * tanFov;
            float cam_y = ndc_y * tanFov;
            float cam_z = -1.0f; // looking down -Z in camera space

            // world dir = forward*cam_z + right*cam_x + up*cam_y
            vec3s dir;
            dir.x = forward.x * cam_z + right.x * cam_x + up.x * cam_y;
            dir.y = forward.y * cam_z + right.y * cam_x + up.y * cam_y;
            dir.z = forward.z * cam_z + right.z * cam_x + up.z * cam_y;

            vec3s_normalize_inplace(&dir);
            dirs[idx] = dir;
        }
    }

    return dirs;
}

// ------------------------------------------------------------
// Single ray march intersection (simple step sampling).
// Returns 1 if hit, 0 otherwise. out_p is the intersection point.
int ht_intersect_heightmap_ray(
    const float* hmap, int hm_w, int hm_l,
    const vec3s origin, const vec3s dir,
    float step, float max_t,
    float* out_t, vec3s* out_p
)
{
    float t = 0.0f;
    while (t < max_t) {
        float px = origin.x + dir.x * t;
        float py = origin.y + dir.y * t;
        float pz = origin.z + dir.z * t;

        // sample height (treat hmap indexed by x across, z down)
        float h = sample_hmap_nearest(hmap, hm_w, hm_l, px, pz);

        // if py <= surface height -> hit
        if (py <= h) {
            if (out_t) *out_t = t;
            if (out_p) { out_p->x = px; out_p->y = py; out_p->z = pz; }
            return 1;
        }

        t += step;
    }
    return 0;
}

// ------------------------------------------------------------
// Trace all rays: allocate out_t (float) and out_points (vec3s).
// out_t[i] = -1.0f if miss else t value. out_points[i] is intersection point (valid only if hit).
// ------------------------------------------------------------
void ht_trace_all(
    const float* hmap, int hm_w, int hm_l,
    const Camera* cam,
    int screenW, int screenH,
    float step, float max_t,
    float** out_t_ptr,
    vec3s** out_points_ptr
)
{
    int total = screenW * screenH;
    float* out_t = (float*) malloc(sizeof(float) * total);
    vec3s* out_p = (vec3s*) malloc(sizeof(vec3s) * total);
    if (!out_t || !out_p) {
        if (out_t) free(out_t);
        if (out_p) free(out_p);
        *out_t_ptr = NULL;
        *out_points_ptr = NULL;
        return;
    }

    // Generate directions on-the-fly (no large additional temp arrays)
    float tanFov = tanf(glm_rad(FOVY) * 0.5f);
    float aspect = (float)screenW / (float)screenH;
    vec3s forward = cam->front;
    vec3s right   = cam->right;
    vec3s up      = cam->up;
    vec3s origin; origin.x = cam->pos.x; origin.y = cam->pos.y; origin.z = cam->pos.z;

    for (int py = 0; py < screenH; ++py) {
        for (int px = 0; px < screenW; ++px) {
            int idx = py * screenW + px;

            float ndc_x = ((px + 0.5f) / (float)screenW) * 2.0f - 1.0f;
            float ndc_y = 1.0f - ((py + 0.5f) / (float)screenH) * 2.0f;

            float cam_x = ndc_x * aspect * tanFov;
            float cam_y = ndc_y * tanFov;
            float cam_z = -1.0f;

            vec3s dir;
            dir.x = forward.x * cam_z + right.x * cam_x + up.x * cam_y;
            dir.y = forward.y * cam_z + right.y * cam_x + up.y * cam_y;
            dir.z = forward.z * cam_z + right.z * cam_x + up.z * cam_y;
            vec3s_normalize_inplace(&dir);

            float tval;
            vec3s hitp;
            int hit = ht_intersect_heightmap_ray(hmap, hm_w, hm_l, origin, dir, step, max_t, &tval, &hitp);
            if (hit) {
                out_t[idx] = tval;
                out_p[idx] = hitp;
            } else {
                out_t[idx] = -1.0f;
                out_p[idx].x = out_p[idx].y = out_p[idx].z = 0.0f;
            }
        }
    }

    *out_t_ptr = out_t;
    *out_points_ptr = out_p;
}
