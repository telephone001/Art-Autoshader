//written by Revanth Krishna Bachu rbachu@wisc.edu
#include "cglm/vec3.h"

double diffuse_intensity(const vec3& normal, const vec3& light_dir) {
    auto n = unit_vector(normal);        
    auto l = unit_vector(light_dir); 
    double cos_theta = dot(n, l); 

    return fmax(0.0, cos_theta); //essentially multiply by the plane's color at that pixel and store it there
}


inline double dot(const vec3& u, const vec3& v) {
    return u.x() * v.x() + u.y() * v.y() + u.z() * v.z();
}

inline vec3 unit_vector(const vec3& v) {
    return v / v.length();   
}
