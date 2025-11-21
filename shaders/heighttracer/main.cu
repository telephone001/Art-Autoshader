// main.cu
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ---------------- Shared Constants ----------------
#define HM_WIDTH   4096
#define HM_HEIGHT  4096
#define MAX_RAY_T  2000.0f
#define STEP_SIZE  0.2f

const size_t HM_SIZE  = (size_t)HM_WIDTH * (size_t)HM_HEIGHT;

// ---------------- Device Code ----------------
__device__ float getHeight(float x, float y, const int* heightmap) {
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);

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
    const size_t  numRays
) {
    size_t rayIndex = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
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
inline void checkCuda(cudaError_t err, const char* ctx = "") {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error [%s]: %s\n", ctx, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template <typename T>
void allocateAndCopy(const T* hostPtr, T** devicePtr, size_t count) {
    size_t size_bytes = count * sizeof(T);
    checkCuda(cudaMalloc((void**)devicePtr, size_bytes), "cudaMalloc");
    checkCuda(cudaMemcpy(*devicePtr, hostPtr, size_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H->D");
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

std::vector<float3> initializeRaysOrigins(size_t totalRays) {
    std::vector<float3> rays(totalRays);
    // Map each ray to a grid sample
    for (size_t i = 0; i < totalRays; ++i) {
        int x = (int)(i % HM_WIDTH);
        int y = (int)(i / HM_WIDTH);
        rays[i] = make_float3((float)x, (float)y, 50.0f);
    }
    return rays;
}

std::vector<float3> initializeRaysDirections(size_t totalRays) {
    std::vector<float3> dirs(totalRays);
    for (size_t i = 0; i < totalRays; ++i) {
        dirs[i] = make_float3(0.0f, 0.0f, -1.0f);
    }
    return dirs;
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
        size_t total_rays = HM_SIZE; // 4096*4096
        printf("Total rays (GPU): %zu\n", total_rays);

        std::vector<float3>  h_origins    = initializeRaysOrigins(total_rays);
        std::vector<float3>  h_directions = initializeRaysDirections(total_rays);
        std::vector<float>   h_t_results(total_rays);
        std::vector<float3>  h_point_results(total_rays);

        // ---- Copy to Device ----
        allocateAndCopy(h_heightmap.data(), &d_heightmap, HM_SIZE);
        allocateAndCopy(h_origins.data(),   &d_origins,   total_rays);
        allocateAndCopy(h_directions.data(), &d_directions, total_rays);

        checkCuda(cudaMalloc((void**)&d_t_results,     total_rays * sizeof(float)), "d_t_results");
        checkCuda(cudaMalloc((void**)&d_point_results, total_rays * sizeof(float3)), "d_point_results");

        // ------------------------------------
        // GPU TIMER START
        // ------------------------------------
        cudaEvent_t start, stop;
        checkCuda(cudaEventCreate(&start), "create start event");
        checkCuda(cudaEventCreate(&stop), "create stop event");
        checkCuda(cudaEventRecord(start), "record start");

        // ---- Launch Kernel ----
        int blockSize = 256;
        size_t gridSize  = (total_rays + blockSize - 1) / blockSize;

        printf("Launching kernel… Grid=%zu Block=%d\n", gridSize, blockSize);

        intersectHeightmap<<<(int)gridSize, blockSize>>>(
            d_origins,
            d_directions,
            d_heightmap,
            d_t_results,
            d_point_results,
            total_rays
        );

        // check kernel launch
        checkCuda(cudaGetLastError(), "kernel launch");

        checkCuda(cudaEventRecord(stop), "record stop");
        checkCuda(cudaEventSynchronize(stop), "synchronize stop");   // <<< END GPU TIMER

        float gpuTime = 0.0f;
        checkCuda(cudaEventElapsedTime(&gpuTime, start, stop), "elapsed time");

        printf("CUDA Kernel Time: %.3f ms\n", gpuTime);

        // ------------------------------------
        // Measure memcpy time (device->host)
        // ------------------------------------
        cudaEvent_t mc_start, mc_stop;
        checkCuda(cudaEventCreate(&mc_start), "create mc_start");
        checkCuda(cudaEventCreate(&mc_stop), "create mc_stop");
        checkCuda(cudaEventRecord(mc_start), "record mc_start");

        checkCuda(cudaMemcpy(h_t_results.data(), d_t_results, total_rays * sizeof(float), cudaMemcpyDeviceToHost), "memcpy t");
        checkCuda(cudaMemcpy(h_point_results.data(), d_point_results, total_rays * sizeof(float3), cudaMemcpyDeviceToHost), "memcpy points");

        checkCuda(cudaEventRecord(mc_stop), "record mc_stop");
        checkCuda(cudaEventSynchronize(mc_stop), "sync mc_stop");
        float memcpyTime = 0.0f;
        checkCuda(cudaEventElapsedTime(&memcpyTime, mc_start, mc_stop), "elapsed memcpy");

        printf("Memcpy Device->Host Time: %.3f ms\n", memcpyTime);

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

        // destroy events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaEventDestroy(mc_start);
        cudaEventDestroy(mc_stop);

    } catch (const std::exception& e) {
        printf("Execution error: %s\n", e.what());
    }

    // Cleanup
    if (d_heightmap)     cudaFree(d_heightmap);
    if (d_origins)       cudaFree(d_origins);
    if (d_directions)    cudaFree(d_directions);
    if (d_t_results)     cudaFree(d_t_results);
    if (d_point_results) cudaFree(d_point_results);

    printf("Simulation finished. GPU memory freed.\n");

    printf("Press Enter to exit...");
    getchar();

    return 0;
}
