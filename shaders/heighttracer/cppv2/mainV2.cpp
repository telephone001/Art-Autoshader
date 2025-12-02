#include <iostream>
#include <cmath>
#include <chrono>

// ------------------------------------------------------
// Access editor heightmap
// ------------------------------------------------------
#include "Art-Autoshader/src/editor.h"   // <-- correct relative path

extern Editor editor;              // Provided by engine


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
// Ray–Heightmap Intersection
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
// MAIN V2
// ------------------------------------------------------
int main()
{
    using namespace std::chrono;

    std::cout << "=== CPU Heightmap Tracer V2 ===\n";

    //----------------------------------------------------------------------
    // 1. Access heightmap from editor (THIS IS YOUR REAL DATA)
    //----------------------------------------------------------------------
    int width  = editor.hmap_w;
    int height = editor.hmap_l;
    float* heightmap = editor.hmap;

    if (heightmap == nullptr) {
        std::cout << "ERROR: editor heightmap is null!\n";
        return -1;
    }

    std::cout << "Heightmap loaded: "
              << width << " x " << height << "\n";


    //----------------------------------------------------------------------
    // 2. Build a test ray (pointing downward from camera)
    //----------------------------------------------------------------------
    Ray r;

    r.ox = width * 0.5f;   // middle of map
    r.oz = height * 0.5f;
    r.oy = 60.0f;          // camera height

    r.dx = 0.0f;
    r.dy = -1.0f;          // straight down
    r.dz = 0.0f;


    //----------------------------------------------------------------------
    // 3. Ray Trace (CPU)
    //----------------------------------------------------------------------
    float hit_t, hx, hy, hz;

    auto start = high_resolution_clock::now();

    int hit = intersect_heightmap(
        heightmap,
        width,
        height,
        r,
        0.1f,     // step size
        500.0f,   // max distance
        hit_t, hx, hy, hz
    );

    auto end = high_resolution_clock::now();
    double ms = duration<double, std::milli>(end - start).count();


    //----------------------------------------------------------------------
    // 4. Output
    //----------------------------------------------------------------------
    if (hit) {
        std::cout << "HIT at t=" << hit_t
                  << "  (" << hx << ", " << hy << ", " << hz << ")\n";
    } else {
        std::cout << "MISS\n";
    }

    std::cout << "CPU Time Used: " << ms << " ms\n";

    return 0;
}
