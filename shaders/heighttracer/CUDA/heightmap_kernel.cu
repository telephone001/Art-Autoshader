// heightmap_kernel.cu — Device-side CUDA code

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "shared_types.h"

// ---------------- Device Helpers ----------------
__device__ float getHeight(float x, float y, const int* heightmap) {
    int x0 = (int)x;
    int y0 = (int)y;

    if (x0 < 0 || x0 >= HM_WIDTH - 1 || y0 < 0 || y0 >= HM_HEIGHT - 1)
        return -MAX_RAY_T;

    float tx = x - x0;
    float ty = y - y0;

    int idx00 = y0 * HM_WIDTH + x0;
    int idx10 = y0 * HM_WIDTH + (x0 + 1);
    int idx01 = (y0 + 1) * HM_WIDTH + x0;
    int idx11 = (y0 + 1) * HM_WIDTH + (x0 + 1);

    float h00 = heightmap[idx00];
    float h10 = heightmap[idx10];
    float h01 = heightmap[idx01];
    float h11 = heightmap[idx11];

    float h_x0 = h00 * (1 - tx) + h10 * tx;
    float h_x1 = h01 * (1 - tx) + h11 * tx;

    return h_x0 * (1 - ty) + h_x1 * ty;
}

// ---------------- Kernel ----------------
__global__ void intersectHeightmap(
    const float3* rayOrigins,
    const float3* rayDirections,
    const int*    heightmap,
    float*        intersectionT,
    float3*       intersectionPoint,
    const int     numRays
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numRays) return;

    const float3 O = rayOrigins[id];
    const float3 D = rayDirections[id];

    float t = 0.0f;
    for (; t < MAX_RAY_T; t += STEP_SIZE) {
        float3 P = make_float3(O.x + t * D.x, O.y + t * D.y, O.z + t * D.z);

        if (P.x < 0 || P.x >= HM_WIDTH - 1 ||
            P.y < 0 || P.y >= HM_HEIGHT - 1)
            break;

        float surfaceZ = getHeight(P.x, P.y, heightmap);

        if (P.z <= surfaceZ) {
            intersectionT[id]     = t;
            intersectionPoint[id] = P;
            return;
        }
    }

    intersectionT[id]     = MAX_RAY_T;
    intersectionPoint[id] = make_float3(0, 0, 0);
}
