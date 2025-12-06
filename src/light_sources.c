#include "light_sources.h"

extern Camera camera;

/// @brief initializes a light source data struct using malloc on the lights array (so we can have as many light sources as we want for debugging)
/// @param light_sources the light sources data you want to initialize
/// @param shader the shader that the light sources data uses to render a lightsource
/// @param point_light_tex the texture of a point light source
/// @param directional_light_tex the texture of a directional light source
/// @return negative value on error.
int light_sources_data_init(LightSourcesData *light_sources, GLuint shader, GLuint point_light_tex, GLuint directional_light_tex)
{
        LightSourcesData lsd = {     
                //set right now           
                .num_lights = 0,
                .lights_length = MAX_LIGHT_SOURCES,
                .lights = malloc(sizeof(LightSource) * MAX_LIGHT_SOURCES),

                // SET BELOW IN THIS FUNCTION
                .rd = {0},
        };

        //TODO REMEMBER TO CHECK ERRORS !! 
        int err = light_sources_rd_init(&lsd.rd, shader, point_light_tex, directional_light_tex);
        ERR_ASSERT_RET((err >= 0), -2, "light sources renderdata could not be created");

        *light_sources = lsd;

        return 0;
}

/// @brief renders all the lightsources. It will bind all the textures and then render light source one by one (TODO: would be better with instancing)
/// @param light_sources the struct containing all light sources
/// @param projection the projection matrix
/// @param view the view matrix
/// @return 0 on error
int light_sources_render(LightSourcesData *light_sources, mat4 projection, mat4 view)
{
    // --- State Setup for Billboards ---

    // 1. Enable Blending for transparency of the light marker texture
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // 2. IMPORTANT FIX: Prevent writing to the depth buffer (Z-Buffer).
    // This stops the markers from corrupting depth and causing the red block issue.
    glDepthMask(GL_FALSE);

    glUseProgram(light_sources->rd.shader);
    glBindVertexArray(light_sources->rd.vao);

    //These uniforms remain the same for all light sources
    glUniform3fv(
            glGetUniformLocation(light_sources->rd.shader, "cam_right"), 
            1, 
            camera.right.raw
    );

    glUniform3fv(
            glGetUniformLocation(light_sources->rd.shader, "cam_up"), 
            1, 
            camera.up.raw
    );

    glUniform3fv(
            glGetUniformLocation(light_sources->rd.shader, "cam_pos"), 
            1, 
            camera.pos.raw
    );

    glUniformMatrix4fv(glGetUniformLocation(light_sources->rd.shader, "view"), 1, GL_FALSE, (float*)view);
    glUniformMatrix4fv(glGetUniformLocation(light_sources->rd.shader, "projection"), 1, GL_FALSE, (float*)projection);


    //bind 2 texture units
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, light_sources->rd.textures[POINT - 1]);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, light_sources->rd.textures[DIRECTIONAL - 1]);


    //render all light sources 
    for (int i = 0; i < light_sources->num_lights; i++) {
        if (light_sources->lights[i].type == NONE) {
            continue;
        }
        
        glUniform3fv(
                glGetUniformLocation(light_sources->rd.shader, "light_source_pos"), 
                1, 
                light_sources->lights[i].pos.raw
        );

        glUniform1i(
                glGetUniformLocation(light_sources->rd.shader, "light_source_type"), 
                light_sources->lights[i].type      
        );

        glUniform1f(
                glGetUniformLocation(light_sources->rd.shader, "scale"), 
                light_sources->lights[i].render_size      
        );
        
        //a quad is just 2 triangles
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    // --- State Cleanup ---
    // Restore depth mask and disable blending
    glDepthMask(GL_TRUE); 
    glDisable(GL_BLEND);

    return 0;
}


// renderdata for multiple light sources
// NO CHANGES NEEDED IN THIS FUNCTION
/// @brief initializes the renderdata for the light sources struct
/// @param light_sources_rd the light sources renderdata
/// @param shader the shader for the light sources renderdata
/// @param point_light_tex the texture handle for point lights
/// @param directional_light_tex the texture handle for directional lights
/// @return positive vals on success. negative values for fail
int light_sources_rd_init(RenderData *light_sources_rd, GLuint shader, GLuint point_light_tex, GLuint directional_light_tex) 
{
        // Remember that light sources doesn't malloc for vertices. vertices is static memory.
        *light_sources_rd = (RenderData) {
                
                // will be filled out below
                .vao = 0, 
                .vbo = 0, 
                .ebo = 0, 

                // NO texture coords because the texture coords are the same as the quad position.
                .vertices = (float[]){
                        -1.0, -1.0,
                        -1.0,  1.0,
                         1.0, -1.0,
                        
                         1.0,  1.0,
                        -1.0,  1.0,
                         1.0, -1.0,
                }, 
                .vertices_stride = 2,  
                .vertices_length = 2 * 6,

                .indices = NULL, // NO INDICES
                .indices_stride = -1,
                .indices_length = -1,

                .textures = malloc(2 * sizeof(GLuint)), //you gotta alloc memory so that the menu image isn't the same as the projected image 
                .num_textures = 2,   

                .primitive_type = GL_TRIANGLES,
                .shader = shader,
        };


        light_sources_rd->textures[0] = point_light_tex;
        light_sources_rd->textures[1] = directional_light_tex;

        //shader has to be on when doing the texture stuff
        glUseProgram(shader);

        // configure textures in gpu
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, point_light_tex);
        glUniform1i(glGetUniformLocation(shader, "tex_point"), 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, directional_light_tex);
        glUniform1i(glGetUniformLocation(shader, "tex_directional"), 1);

        //now do regular opengl initializaiton
        bind_vao_and_vbo(&(light_sources_rd->vao), &(light_sources_rd->vbo), light_sources_rd->vertices, sizeof(float) * light_sources_rd->vertices_length, GL_STATIC_DRAW);
        
        ERR_ASSERT_RET((light_sources_rd->vao != 0), -3, "vao failed");
        ERR_ASSERT_RET((light_sources_rd->vbo != 0), -4, "vbo failed");


        //we dont send texture coords because the texture coords are the same as the quad position.
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        

        return 0;
}

/// @brief adds a light source to the light sources data
/// @param light_sources_data the struct containing information on all light sources
/// @param light_source a light source that you want to add
/// @return positive values on success. Negative on error
int light_source_add(LightSourcesData *light_sources_data, LightSource light_source)
{
        if (light_sources_data->num_lights >= MAX_LIGHT_SOURCES) {
                return -1;
        }

        

        // the index to place the new light source
        int idx = -1;

        for (int i = 0; i < MAX_LIGHT_SOURCES; i++) {
                if (light_sources_data->lights[i].type == NONE) {
                        idx = i;
                        break;
                }
        }

        ERR_ASSERT_RET((idx >= 0), -2, "Cannot find a spot in the light source array to put a new light source\n");

        light_sources_data->lights[idx] = light_source;
        light_sources_data->num_lights++;

        return 0;
}