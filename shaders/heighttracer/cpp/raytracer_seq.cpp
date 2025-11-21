// main.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>

struct HitResult {
    bool hit;
    float t;
    float px, py, pz;
};

int main() {
    using namespace std::chrono;

    // -------------------------------
    // 1. Scene / grid configuration
    // -------------------------------
    const int width  = 4096;
    const int height = 4096;
    const long long total_rays = (long long)width * (long long)height; // use 64-bit for safety

    const float cellSize = 4.0f;

    const float cam_x = 0.0f;
    const float cam_y = 0.0f;
    const float cam_z = 50.0f;

    std::cout << "Sequential CPU raytrace: " << width << " x " << height
              << " = " << total_rays << " rays\n";

    // -------------------------------
    // 2. Fake heightmap (flat)
    // -------------------------------
    // single-precision floats: 4096*4096 ≈ 16.7M floats ≈ 64 MB
    std::vector<float> heightmap;
    try {
        heightmap.assign((size_t)total_rays, 0.0f);
    } catch (const std::bad_alloc& e) {
        std::cerr << "Failed to allocate heightmap: " << e.what() << "\n";
        return -1;
    }

    // -------------------------------
    // 3. Storage for results
    // -------------------------------
    std::vector<HitResult> results;
    try {
        results.resize((size_t)total_rays);
    } catch (const std::bad_alloc& e) {
        std::cerr << "Failed to allocate results: " << e.what() << "\n";
        return -1;
    }

    // -------------------------------
    // 4. Start CPU timer
    // -------------------------------
    auto cpu_start = high_resolution_clock::now();

    // -------------------------------
    // 5. Sequential ray loop
    // -------------------------------
    for (long long i = 0; i < total_rays; ++i) {
        int gx = (int)(i % width);
        int gy = (int)(i / width);

        float world_x = gx * cellSize;
        float world_y = gy * cellSize;

        float ox = cam_x + world_x;
        float oy = cam_y + world_y;
        float oz = cam_z;

        float dx = 0.0f, dy = 0.0f, dz = -1.0f;

        float terrain_z = heightmap[(size_t)gy * width + gx];

        float t = oz - terrain_z;

        HitResult out;
        if (t > 0.0f) {
            out.hit = true;
            out.t   = t;
            out.px  = ox + t * dx;
            out.py  = oy + t * dy;
            out.pz  = oz + t * dz;
        } else {
            out.hit = false;
            out.t   = -1;
            out.px = out.py = out.pz = 0;
        }

        results[(size_t)i] = out;
    }

    // -------------------------------
    // 6. End CPU timer + print
    // -------------------------------
    auto cpu_end = high_resolution_clock::now();
    double cpu_ms = duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU Time Used: " << std::fixed << std::setprecision(3) << cpu_ms << " ms\n";

    // Print first 10 rays
    std::cout << "\n--- Intersection Results (First 10 Rays) ---\n";
    for (int i = 0; i < 10; ++i) {
        const auto& r = results[i];
        if (r.hit) {
            std::cout << "Ray " << i
                      << ": HIT  t=" << r.t
                      << "  P=(" << r.px << ", "
                                << r.py << ", "
                                << r.pz << ")\n";
        } else {
            std::cout << "Ray " << i << ": MISS\n";
        }
    }

    std::cout << "--------------------------------------------\n";
    std::cout << "Done.\n";

    return 0;
}
