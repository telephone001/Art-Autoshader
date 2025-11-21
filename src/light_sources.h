
#ifndef __LIGHT_SOURCES_H__
#define __LIGHT_SOURCES_H__

#include "general/debug.h"
#include "opengl_util.h"
#include "cglm/struct.h"
#include "glfw_window.h"


#define MAX_LIGHT_SOURCES 100

typedef enum LightType {
        NONE,           //this is only here to check if a light source entry in the array is valid
        POINT,
        DIRECTIONAL,
} LightType;

// add more parameters
typedef struct LightSource {
        LightType type;
        vec3s pos;
} LightSource;

enum LightSourcesVertexAttributes {
	VERTICES 	  = 0,
	POSITIONS  	  = 1,
	NUM_VERTEX_ATTRIBUTES = 2,
};

typedef struct LightSourcesData {
        RenderData rd;                         // renderdata for all the lights
        int num_lights;                 // counts how many light sources are currently there
        LightSource lights[MAX_LIGHT_SOURCES]; // contains all the lights in the program. 
} LightSourcesData;






/// @brief renders all the lightsources. It will bind all the textures and then render light source one by one (TODO: would be better with instancing)
/// @param light_sources the struct containing all light sources
/// @return 0 on error
int light_sources_render(LightSourcesData *light_sources);

/// @brief initializes the renderdata for the light sources struct
/// @param light_sources_rd the light sources renderdata
/// @param shader the shader for the light sources renderdata
/// @param point_light_tex the texture handle for point lights
/// @param directional_light_tex the texture handle for directional lights
/// @return positive vals on success. negative values for fail
int light_sources_rd_init(RenderData *light_sources_rd, GLuint shader, GLuint point_light_tex, GLuint directional_light_tex);



/// @brief adds a light source to the light sources data
/// @param light_sources_data the struct containing information on all light sources
/// @param light_source a light source that you want to add
/// @return positive values on success. Negative on error
int light_source_add(LightSourcesData *light_sources_data, LightSource light_source);









#endif