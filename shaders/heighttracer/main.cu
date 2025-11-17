#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ---------------- Shared Constants ----------------
#define HM_WIDTH   1024
#define HM_HEIGHT  1024
#define MAX_RAY_T  2000.0f   // Max ray distance
#define STEP_SIZE  0.2f      // Ray-march step length

const int NUM_RAYS = 256;
const int HM_SIZE  = HM_WIDTH * HM_HEIGHT;

// ---------------- Device Code ----------------
__device__ float getHeight(float x, float y, const int* heightmap) {
    int x0 = (int)x;
    int y0 = (int)y;

    if (x0 < 0 || x0 >= HM_WIDTH - 1 || y0 < 0 || y0 >= HM_HEIGHT - 1)
        return -MAX_RAY_T;

    float tx = x - (float)x0;
    float ty = y - (float)y0;

    int index00 = y0 * HM_WIDTH + x0;
    int index10 = y0 * HM_WIDTH + (x0 + 1);
    int index01 = (y0 + 1) * HM_WIDTH + x0;
    int index11 = (y0 + 1) * HM_WIDTH + (x0 + 1);

    float h00 = (float)heightmap[index00];
    float h10 = (float)heightmap[index10];
    float h01 = (float)heightmap[index01];
    float h11 = (float)heightmap[index11];

    float h_x0 = h00 * (1.0f - tx) + h10 * tx;
    float h_x1 = h01 * (1.0f - tx) + h11 * tx;
    return h_x0 * (1.0f - ty) + h_x1 * ty;
}

__global__ void intersectHeightmap(
    const float3* rayOrigins,
    const float3* rayDirections,
    const int*    heightmap,
    float*        intersectionT,
    float3*       intersectionPoint,
    const int     numRays
) {
    int rayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (rayIndex >= numRays) return;

    const float3 O = rayOrigins[rayIndex];
    const float3 D = rayDirections[rayIndex];

    float t = 0.0f;
    for (; t < MAX_RAY_T; t += STEP_SIZE) {
        float3 P = make_float3(O.x + t * D.x, O.y + t * D.y, O.z + t * D.z);

        if (P.x < 0.0f || P.x >= (float)(HM_WIDTH - 1) ||
            P.y < 0.0f || P.y >= (float)(HM_HEIGHT - 1))
            break;

        float surfaceZ = getHeight(P.x, P.y, heightmap);

        if (P.z <= surfaceZ) {
            intersectionT[rayIndex] = t;
            intersectionPoint[rayIndex] = P;
            return;
        }
    }

    intersectionT[rayIndex] = MAX_RAY_T;
    intersectionPoint[rayIndex] = make_float3(0.0f, 0.0f, 0.0f);
}

// ---------------- Host Helpers ----------------
template <typename T>
void allocateAndCopy(const T* hostPtr, T** devicePtr, size_t count) {
    size_t size_bytes = count * sizeof(T);
    cudaMalloc((void**)devicePtr, size_bytes);
    cudaMemcpy(*devicePtr, hostPtr, size_bytes, cudaMemcpyHostToDevice);
}

std::vector<int> initializeHeightmap() {
    std::vector<int> heightmap(HM_SIZE);
    float centerX = HM_WIDTH / 2.0f;
    float centerY = HM_HEIGHT / 2.0f;

    for (int y = 0; y < HM_HEIGHT; ++y) {
        for (int x = 0; x < HM_WIDTH; ++x) {
            float dx = x - centerX;
            float dy = y - centerY;
            float dist = std::sqrt(dx * dx + dy * dy);
            float h = 20.0f * std::exp(-(dist / 250.0f) * (dist / 250.0f));
            heightmap[y * HM_WIDTH + x] = static_cast<int>(h);
        }
    }
    return heightmap;
}

std::vector<float3> initializeRays(bool isOrigin) {
    std::vector<float3> rays(NUM_RAYS);
    for (int i = 0; i < NUM_RAYS; ++i) {
        if (isOrigin) {
            rays[i] = make_float3(
                static_cast<float>(i * (HM_WIDTH / NUM_RAYS)),
                static_cast<float>(i * (HM_HEIGHT / NUM_RAYS)) * 0.5f,
                50.0f
            );
        } else {
            rays[i] = make_float3(0.0f, 0.0f, -1.0f);
        }
    }
    return rays;
}

// ---------------- Main Function ----------------
int main() {
    int*    d_heightmap      = nullptr;
    float3* d_origins        = nullptr;
    float3* d_directions     = nullptr;
    float*  d_t_results      = nullptr;
    float3* d_point_results  = nullptr;

    try {
        printf("Starting CUDA Heightmap Raytracing Simulation...\n");

        // ---- Host Data ----
        std::vector<int>     h_heightmap  = initializeHeightmap();
        std::vector<float3>  h_origins    = initializeRays(true);
        std::vector<float3>  h_directions = initializeRays(false);
        std::vector<float>   h_t_results(NUM_RAYS);
        std::vector<float3>  h_point_results(NUM_RAYS);

        // ---- Copy to Device ----
        allocateAndCopy(h_heightmap.data(), &d_heightmap, HM_SIZE);
        allocateAndCopy(h_origins.data(),   &d_origins,   NUM_RAYS);
        allocateAndCopy(h_directions.data(), &d_directions, NUM_RAYS);

        cudaMalloc((void**)&d_t_results,     NUM_RAYS * sizeof(float));
        cudaMalloc((void**)&d_point_results, NUM_RAYS * sizeof(float3));

        // ---- Launch Kernel ----
        int blockSize = 256;
        int gridSize  = (NUM_RAYS + blockSize - 1) / blockSize;

        printf("Launching kernel… Grid=%d Block=%d\n", gridSize, blockSize);

        intersectHeightmap<<<gridSize, blockSize>>>(
            d_origins,
            d_directions,
            d_heightmap,
            d_t_results,
            d_point_results,
            NUM_RAYS
        );

        cudaDeviceSynchronize();

        // ---- Copy Back Results ----
        cudaMemcpy(h_t_results.data(), d_t_results, NUM_RAYS * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_point_results.data(), d_point_results, NUM_RAYS * sizeof(float3), cudaMemcpyDeviceToHost);

        // ---- Print First 10 Rays ----
        printf("\n--- Intersection Results (First 10 Rays) ---\n");
        for (int i = 0; i < 10; ++i) {
            if (h_t_results[i] < MAX_RAY_T) {
                printf("Ray %d: HIT  t=%f  P=(%f, %f, %f)\n",
                       i, h_t_results[i],
                       h_point_results[i].x,
                       h_point_results[i].y,
                       h_point_results[i].z);
            } else {
                printf("Ray %d: MISS\n", i);
            }
        }
        printf("------------------------------------------\n");

    } catch (const std::exception& e) {
        printf("Execution error: %s\n", e.what());
    }

    // ---- Cleanup ----
    if (d_heightmap)     cudaFree(d_heightmap);
    if (d_origins)       cudaFree(d_origins);
    if (d_directions)    cudaFree(d_directions);
    if (d_t_results)     cudaFree(d_t_results);
    if (d_point_results) cudaFree(d_point_results);

    printf("Simulation finished. GPU memory freed.\n");

    // ---- Pause so console stays open ----
    printf("Press Enter to exit...");
    getchar();

    return 0;
}
