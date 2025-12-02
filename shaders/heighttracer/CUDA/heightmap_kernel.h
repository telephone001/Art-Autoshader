#pragma once
#include <cuda_runtime.h>

__global__ void intersectHeightmap(
    const float3* rayOrigins,
    const float3* rayDirections,
    const int*    heightmap,
    float*        intersectionT,
    float3*       intersectionPoint,
    const int     numRays
);
