#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// ----------------------
// Ray Struct
// ----------------------
struct Ray {
    float ox, oy, oz;   // origin
    float dx, dy, dz;   // direction (normalized)
};

// ----------------------
// Sample Height (nearest)
// ----------------------
static inline float sample_heightmap(
    const float* hmap,
    int width,
    int height,
    float x,
    float z
) {
    int ix = (int)x;
    int iz = (int)z;

    if (ix < 0 || iz < 0 || ix >= width || iz >= height)
        return -INFINITY;

    return hmap[iz * width + ix];
}

// ------------------------------------------------------
// Ray–Heightmap Intersection (CPU)
// Returns 1 on hit, 0 on miss
// ------------------------------------------------------
int intersect_heightmap(
    const float* hmap,
    int width,
    int height,
    const Ray& ray,
    float step,
    float max_t,
    float& out_t,
    float& out_x,
    float& out_y,
    float& out_z
) {
    float t = 0.0f;

    while (t < max_t)
    {
        float x = ray.ox + t * ray.dx;
        float y = ray.oy + t * ray.dy;
        float z = ray.oz + t * ray.dz;

        float terrain_y = sample_heightmap(hmap, width, height, x, z);

        if (terrain_y == -INFINITY) {
            t += step;
            continue;
        }

        if (y <= terrain_y) {
            out_t = t;
            out_x = x;
            out_y = y;
            out_z = z;
            return 1;
        }

        t += step;
    }

    return 0;
}

// ------------------------------------------------------
// MAIN
// ------------------------------------------------------
int main() {
    using namespace std::chrono;

    //----------------------------------------------------
    // 1. Create Example Heightmap (You will replace this)
    //----------------------------------------------------
    const int width  = 256;
    const int height = 256;

    std::vector<float> heightmap(width * height);

    // simple fake terrain (replace with real data)
    for (int z = 0; z < height; z++) {
        for (int x = 0; x < width; x++) {
            float dx = x - width / 2;
            float dz = z - height / 2;
            float dist = sqrt(dx * dx + dz * dz);
            heightmap[z * width + x] = 10.0f * exp(-(dist / 80.0f));
        }
    }

    //----------------------------------------------------
    // 2. Launch rays
    //----------------------------------------------------
    Ray r;
    r.ox = 128;
    r.oz = 128;
    r.oy = 60;         // camera height
    r.dx = 0;
    r.dz = 0;
    r.dy = -1;         // pointing downward

    float hit_t, hx, hy, hz;

    auto start = high_resolution_clock::now();

    int hit = intersect_heightmap(
        heightmap.data(),
        width,
        height,
        r,
        0.1f,        // step
        500.0f,      // max_t
        hit_t, hx, hy, hz
    );

    auto end = high_resolution_clock::now();
    double ms = duration<double, std::milli>(end - start).count();

    //----------------------------------------------------
    // 3. Print results
    //----------------------------------------------------
    if (hit) {
        std::cout << "HIT at t=" << hit_t
                  << "  (" << hx << ", " << hy << ", " << hz << ")\n";
    } else {
        std::cout << "MISS\n";
    }

    std::cout << "CPU Time Used: " << ms << " ms\n";

    return 0;
}
