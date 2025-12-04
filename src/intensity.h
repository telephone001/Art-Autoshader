#include "cglm/vec3.h"

double diffuse_intensity(const vec3& normal, const vec3& light_dir);

inline double dot(const vec3& u, const vec3& v);

inline vec3 unit_vector(const vec3& v);