#include <stdio.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_include.h>


#include "src/gui.h"
#include "src/editor.h"
#include "src/light_sources.h"
#include "src/heighttracer_cpu.h"

#include "general/debug.h"


#include "opengl_util.h"
#include "opengl_texture_util.h"
#include "glfw_window.h"

#include <cglm/mat4.h>
#include <cglm/cam.h>
#include <cglm/util.h>
#include <cglm/struct.h>
#include <cglm/io.h>

#ifdef __CUDACC__

#include "src/heighttracer_cpu.h"
#include "src/heighttracer_cuda.cuh"

#endif



#define SCR_LENGTH 800
#define SCR_HEIGHT 800

#define MAX_EDITORS 100

#define ORTHO_SCALE 1.0/100.0

extern Camera camera; //handler for cameradata;
extern int in_menu;   //menu status


typedef enum DebugThing {
        DEBUG_NONE,
        SPAWN_EDITOR,
        DELETE_LIGHTS,
        SPAWN_POINT_LIGHT,
        SPAWN_DIRECTIONAL_LIGHT,
        SPAWN_LIGHTS_FOR_CAMERA_RAYS,
        SPAWN_LIGHTS_FOR_INTERSECTIONS,
        CPU_TEST_RAYTRACE,
        CUDA_TEST_RAYTRACE
} DebugThing;

DebugThing debug_thing = 0; // TODO  REMOVE IN FINAL PRODUCT

//TODO these are only in global because I dont want to send a window hint pointer to callback

mat4 flycam_projection;
mat4 flycam_view;

mat4 offset_view;
mat4 ortho_proj; //used to actually edit the heightmap


// Convenience macros for edbert
#define V3_X(v) ((v).raw[0])
#define V3_Y(v) ((v).raw[1])
#define V3_Z(v) ((v).raw[2])
#define V2_X(v) ((v).raw[0])
#define V2_Y(v) ((v).raw[1])

/// @brief 
void opengl_settings_init()
{
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);

        //static camera data
	glm_perspective(glm_rad(FOVY), (float)SCR_LENGTH / SCR_HEIGHT, NEAR, FAR, flycam_projection);

        glm_ortho(
                -SCR_LENGTH * ORTHO_SCALE, //left
                SCR_LENGTH * ORTHO_SCALE, //right
                -SCR_HEIGHT * ORTHO_SCALE, //bottom
                SCR_HEIGHT * ORTHO_SCALE, //top
                NEAR, //near
                FAR,  //far
                ortho_proj
        );

        //set background color
        glClearColor(0.1, 0.2, 0.3, 1.0);

}

/// @brief      This handles how switching between menu and screen happens using esc key. 
///
///             The only key callback you should use for this program. This should be set as the keyback
///             only AFTER the gui and the glfw window have been initialized in order to overwrite 
///             the callbacks they brought. 
///             
/// @param wnd the glfw window
/// @param key the key that has done an action
/// @param scancode hardware code for the key
/// @param action what action the key has done
/// @param mods bitfield indicating what modifiers were on the keys
void key_callback_menu_switching(
        GLFWwindow *wnd, 
        int key, 
        int scancode, 
        int action, 
        int mods)
{

        if (in_menu) {
                //let the nuklear menu use their callbacks
                glfwSetScrollCallback(wnd, nk_gflw3_scroll_callback);
                glfwSetCharCallback(wnd, nk_glfw3_char_callback);
                glfwSetMouseButtonCallback(wnd, nk_glfw3_mouse_button_callback);
                

                //this will do the nuklear key callback
                nk_glfw3_key_callback(wnd, key, scancode, action, mods);

        } else {
                //give back mouse control
	        glfwSetCursorPosCallback(wnd, mouse_callback);

                //this can be changed to the normal callbacks
                glfwSetScrollCallback(wnd, NULL);
                glfwSetCharCallback(wnd, NULL);
                glfwSetMouseButtonCallback(wnd, NULL);

                //do some overwriting
                if (key == GLFW_KEY_LEFT_SHIFT) {
	        	switch (action) {
	        	case GLFW_PRESS:
	        		camera.speed = NORMAL_CAMERA_SPEED / 5;
	        		break;
                        
	        	case GLFW_RELEASE:
	        		camera.speed = NORMAL_CAMERA_SPEED;
	        		break;
	        	}
	        }
                
	        if (key == GLFW_KEY_LEFT_CONTROL) {
	        	switch (action) {
	        	case GLFW_PRESS:
	        		camera.speed = NORMAL_CAMERA_SPEED * 3;
	        		break;
                        
	        	case GLFW_RELEASE:
	        		camera.speed = NORMAL_CAMERA_SPEED;
	        		break;
	        	}
	        }

                if (key == GLFW_KEY_P && action == GLFW_PRESS) debug_thing = SPAWN_EDITOR;
                if (key == GLFW_KEY_L && action == GLFW_PRESS) debug_thing = DELETE_LIGHTS;
                if (key == GLFW_KEY_1 && action == GLFW_PRESS) debug_thing = SPAWN_POINT_LIGHT;
                if (key == GLFW_KEY_2 && action == GLFW_PRESS) debug_thing = SPAWN_DIRECTIONAL_LIGHT;
                if (key == GLFW_KEY_5 && action == GLFW_PRESS) debug_thing = CPU_TEST_RAYTRACE;
                if (key == GLFW_KEY_6 && action == GLFW_PRESS) debug_thing = CUDA_TEST_RAYTRACE;
	}

        //these are required because the only other alternative would be global variables.
        // you cannot recieve these as a parameter because they mess up the callback function parameters
        static double last_x_pos, last_y_pos;

        // in addition to the nuklear callback, I will overwrite their esc functionality
        // so the program can exit gui mode using esc
        if (key == GLFW_KEY_ESCAPE && action != GLFW_PRESS) {
	        if (in_menu) {
	        	glfwSetInputMode(wnd, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	        	glfwSetCursorPos(wnd, last_x_pos, last_y_pos);
	        } else {
	        	glfwGetCursorPos(wnd, &last_x_pos, &last_y_pos);
	        	glfwSetInputMode(wnd, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	        }

	        in_menu = !in_menu;
	}

}


/// @brief will make sure the viewport matches the new dimensions when resized and updates the projection matrix
/// @param window window of glfw program
/// @param width width of screen
/// @param height height of screen
void framebuffer_size_callback(GLFWwindow *const window, int width, int height)
{
	//change the projection matrix
	glm_perspective(glm_rad(FOVY), (double)width / (double)height, NEAR, FAR, flycam_projection);

        glm_ortho(
                -width * ORTHO_SCALE, //left
                width * ORTHO_SCALE, //right
                -height * ORTHO_SCALE, //bottom
                height * ORTHO_SCALE, //top
                NEAR, //near
                FAR,  //far
                ortho_proj
        );

	//resize the viewport
	glViewport(0, 0, width, height);
}



int main() 
{
        if (!glfwInit()) return -1;
        

        GLFWwindow* wnd = glfw_setup(3, 3, SCR_LENGTH, SCR_HEIGHT, "art autoshader");
	    if (!wnd)
	    	return -1;

        opengl_settings_init();

        GLuint shader_cam_proj = create_shader_program(
                "shaders/cam_proj.vert",
                "shaders/cam_proj.frag", 
                NULL, 
                NULL, 
                NULL
        );

        GLuint shader_cam_plane = create_shader_program(
                "shaders/cam_plane.vert",
                "shaders/cam_plane.frag", 
                NULL, 
                NULL, 
                NULL
        );

        GLuint shader_hmap = create_shader_program(
                "shaders/hmap.vert",
                "shaders/hmap.frag", 
                NULL, 
                NULL, 
                NULL
        );	

        GLuint shader_light_sources = create_shader_program(
                "shaders/light_source.vert",
                "shaders/light_source.frag", 
                NULL, 
                NULL, 
                NULL
        );	


        MenuOptions gui_menu;
        int err = nuklear_menu_init(&gui_menu, wnd, "fonts/american-typewriter.ttf", 22); 
        ERR_ASSERT_RET((err >= 0), -2, "nuklear window could not be created");

        // this is required AFTER nuklear_menu_init because it uses the callbacks
        glfwSetKeyCallback(wnd, key_callback_menu_switching);



        GLuint point_light_tex, directional_light_tex;
        load_2dtexture(&point_light_tex, "textures/point_light.jpg", GL_RGB);
        load_2dtexture(&directional_light_tex, "textures/directional_light.jpg", GL_RGB);

        LightSourcesData light_sources_data = {0};
        err = light_sources_data_init(&light_sources_data, shader_light_sources, point_light_tex, directional_light_tex);
        ERR_ASSERT_RET((err >= 0), -2, "light sources struct could not be created");


        // THIS IS FOR DEBUGGING
        Editor editors[MAX_EDITORS] = {0};
        int edit_idx = 0; // if we were to add another editor, at what idx will it be added in editors

        glfwSetInputMode(wnd, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        GL_PRINT_ERR();
        
        while (!glfwWindowShouldClose(wnd)) {
                //GL_PRINT_ERR();
                static float last_frame;
                float t = glfwGetTime();
		float delta_time = t - last_frame;
		last_frame = t;	
                

                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                //edit camera to look in correct direction
		get_cam_view(camera, flycam_view);
		

                // If debug thing is 1, we make an editor
                if (debug_thing == SPAWN_EDITOR) {
                        int err = editor_init(
                                &(editors[edit_idx]), 
                                wnd, 
                                &gui_menu, 
                                shader_cam_proj, 
                                shader_cam_plane,
                                shader_hmap,
                                100, // PLACHOLDER
                                100  // PLACHOLDER
                        );
                        
                        hmap_edit_zero(&(editors[edit_idx]));


                        if (err < 0) {
                                //Required before editor_free if gui_menu is using the texture.
                                if (editors[edit_idx].mdl_cam_plane.textures != NULL &&
                                    gui_menu.img_tex == editors[edit_idx].mdl_cam_plane.textures[0]) {
                                        editors[edit_idx].mdl_cam_plane.textures[0] = 0;
                                }
                                editor_free_forced(&(editors[edit_idx]), 0);
                        } else {
                                edit_idx++;
                        }

                        if (edit_idx >= MAX_EDITORS) {
                                edit_idx = 0;
                        }

                        debug_thing = DEBUG_NONE; 
                }

                // delete editor AND light sources
                if (debug_thing == DELETE_LIGHTS) {

                        for (int i = 0; i < MAX_LIGHT_SOURCES; i++) {
                                light_sources_data.lights[i] = (LightSource){0};
                        }

                        debug_thing = DEBUG_NONE;
                }

                if (debug_thing == SPAWN_POINT_LIGHT) {
                        //TODO check this error
                        light_source_add(&light_sources_data, (LightSource){POINT, camera.pos, 0.5});
                        debug_thing = DEBUG_NONE;
                }

                if (debug_thing == SPAWN_DIRECTIONAL_LIGHT) {
                        light_source_add(&light_sources_data, (LightSource){DIRECTIONAL, camera.pos, 1});
                        debug_thing = DEBUG_NONE;
                }
                
                // CPU raytrace
                if (debug_thing == CPU_TEST_RAYTRACE && editors[0].mdl_cam_proj.vao != 0) {
                        int width, height;
                        glfwGetWindowSize(wnd, &width, &height);
                        vec3s* cam_dirs = ht_generate_camera_directions(&camera, width, height);
    
                        // invert the hmap transformation
                        mat4s inv_trans;
                        glm_mat4_inv(editors[gui_menu.which_editor_selected].hmap_transform.matrix, inv_trans.raw);

                        mat4s trans;
                        glm_mat4_copy(editors[gui_menu.which_editor_selected].hmap_transform.matrix, trans.raw);


                        vec3s translated_cam_pos = glms_mat4_mulv3(inv_trans, camera.pos, 1);



                        double start_time = glfwGetTime(); // Start CPU timer
                        int lights_added_cpu = 0;
    
                        for (int i = 0; i < height; i += 25) {
                            for (int j = 0; j < width; j += 25) {
                                float t_ray;
                                vec3s point;

                                // LAST PARAM HAS TO BE 0. TREAT IT AS A DIRECTION
                                vec3s transformed_cam_dirs = glms_mat4_mulv3(inv_trans, cam_dirs[i*width + j], 0);
                                

                                int hit = ht_intersect_heightmap_ray(
                                        editors[gui_menu.which_editor_selected].hmap, 
                                        editors[gui_menu.which_editor_selected].hmap_w, 
                                        editors[gui_menu.which_editor_selected].hmap_l,
                                        translated_cam_pos, 
                                        transformed_cam_dirs, 
                                        0.1f, 
                                        500.0f, 
                                        &t_ray, 
                                        &point
                                );

                                if (hit) {
                                    //vec3s world_point = glms_vec3_add(camera.pos, glms_vec3_scale(cam_dirs[i * width + j], t_ray));
                                    vec3s world_point = glms_mat4_mulv3(trans, point, 1);
                                    light_source_add(&light_sources_data, (LightSource) { POINT, world_point, 1.0f });
                                    //printf("CPU Hit at t=%.6f: %.6f %.6f %.6f\n", t_ray, V3_X(world_point), V3_Y(world_point), V3_Z(world_point));
                                    lights_added_cpu++; // Increment CPU lights count
                                }
                            }
                        }
                        double end_time = glfwGetTime(); // Stop CPU timer
                        printf("CPU Raytracing finished: %d lights added. Time taken: %.6f seconds\n", lights_added_cpu, end_time - start_time);
    
    
                        free(cam_dirs);
                        debug_thing = DEBUG_NONE;
                }

#ifdef __CUDACC__
                // --- CUDA RAYTRACING ---
                if (debug_thing == CUDA_TEST_RAYTRACE && editors[0].mdl_cam_proj.vao != 0) {
                        int width, height;
                        glfwGetWindowSize(wnd, &width, &height);
    
                        float* t_results_cuda = NULL;
                        vec3s* points_results_cuda = NULL;
    
                        // Start CUDA timer (measures host-side call, including transfer and kernel launch)
                        double start_time = glfwGetTime();
    
                        ht_trace_all_cuda(
                            editors[0].hmap, editors[0].hmap_w, editors[0].hmap_l,
                            &camera, width, height, 0.1f, 500.0f,
                            &t_results_cuda, &points_results_cuda
                        );
    
                        double end_time = glfwGetTime(); // Stop CUDA timer
    
                        if (t_results_cuda && points_results_cuda) {
                            int lights_added_cuda = 0; // Counter for lights spawned on host
                            // Step safely over pixels (same step size as CPU version for comparison)
                            int step = 50;
                            if (width < step) step = 1;
                            if (height < step) step = 1;
    
                            for (int i = 0; i < height; i += step) {
                                for (int j = 0; j < width; j += step) {
                                    int idx = i * width + j;
    
                                    // Check for a valid hit (t >= 0.0f is considered a hit from the caller's perspective)
                                    if (t_results_cuda[idx] >= 0.0f) {
                                        vec3s hit_point = points_results_cuda[idx];
                                        float hit_time = t_results_cuda[idx];
    
                                        // Print every sampled hit point (like the CPU version)
                                        fprintf(stdout, "CUDA Hit at t=%.6f: %.6f %.6f %.6f\n",
                                            hit_time, V3_X(hit_point), V3_Y(hit_point), V3_Z(hit_point));
    
                                        // Spawn point light if there is space
                                        if (light_sources_data.num_lights < MAX_LIGHT_SOURCES) {
                                            light_source_add(&light_sources_data, (LightSource) { POINT, hit_point, 1.0f });
                                            lights_added_cuda++; // Increment CUDA lights count
                                        }
                                    }
                                }
                            }
    
                            // Summary print for CUDA
                            printf("CUDA Raytracing finished: %d lights added. Time taken: %.6f seconds\n", lights_added_cuda, end_time - start_time);
    
    
                            free(t_results_cuda);
                            free(points_results_cuda);
                        }
                        else {
                            fprintf(stderr, "CUDA Raytracing failed: no results returned\n");
                        }
    
                        debug_thing = DEBUG_NONE;
                }
#endif

                //Render the editors
                glEnable(GL_DEPTH_TEST);
                glDepthFunc(GL_LESS);
                
                for (int i = 0; i < MAX_EDITORS; i++) {
                        if (editors[i].mdl_cam_proj.vao != 0 ) {
                                editor_render(&(editors[i]), 0, gui_menu.hmap_opacity, flycam_projection, flycam_view);
                        }
                }
                
///////////////////////////////// TESTS ON FRAMEBUFFER ////////////////////////////////////

                glBindFramebuffer(GL_FRAMEBUFFER, gui_menu.ecam_data.fbo);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                
                Camera editor_cam = editors[gui_menu.which_editor_selected].cam;
                float ofx, ofy;
                ofx = gui_menu.ecam_data.pos_offset.x;
                ofy = gui_menu.ecam_data.pos_offset.y;


                vec3s ofr = glms_vec3_scale(editor_cam.right, ofx);
                vec3s ofu = glms_vec3_scale(editor_cam.up, ofy);
                editor_cam.pos = glms_vec3_add(editor_cam.pos, ofr);
                editor_cam.pos = glms_vec3_add(editor_cam.pos, ofu);

                get_cam_view(editor_cam, offset_view); 


                for (int i = 0; i < MAX_EDITORS; i++) {
                        if (editors[i].mdl_cam_proj.vao != 0 ) {
                                editor_render(
                                        &(editors[i]), 
                                        1, 
                                        gui_menu.hmap_opacity,
                                        (gui_menu.ecam_data.in_perspective) ? flycam_projection : ortho_proj, 
                                        offset_view
                                );
                        }
                }

                glBindFramebuffer(GL_FRAMEBUFFER, 0);

///////////////////////////////// TESTS ON FRAMEBUFFER ////////////////////////////////////

                
                // REMEMBER THAT THIS CAN RETURN AN ERROR
                light_sources_render(&light_sources_data, flycam_projection, flycam_view);

///////////////////////////////// GUI_STUFF ////////////////////////////////////   
		if (in_menu) {
                        nuklear_menu_render(wnd, delta_time, &gui_menu);
        		glfwSetCursorPosCallback(wnd, NULL);
        	} else {
                	handle_wasd_move(wnd, delta_time);
                	glfwSetCursorPosCallback(wnd, mouse_callback);
		}

                switch (gui_menu.editor_action) {
                        case EDITOR_ACTION_GOTO:
                                camera = editors[gui_menu.which_editor_selected].cam;
                                gui_menu.editor_action = EDITOR_ACTION_IDLE;
                                break;
                        
                        case EDITOR_ACTION_HMAP_EDIT:
                                Camera ecam = editors[gui_menu.which_editor_selected].cam;
                                float brush_pnt_right = 2 * gui_menu.ecam_data.mouse_offset.x * SCR_LENGTH * ORTHO_SCALE - (SCR_LENGTH * ORTHO_SCALE);
                                float brush_pnt_up = 2 * gui_menu.ecam_data.mouse_offset.y * SCR_HEIGHT * ORTHO_SCALE - (SCR_HEIGHT * ORTHO_SCALE);
                                
                                brush_pnt_right += gui_menu.ecam_data.pos_offset.x;
                                brush_pnt_up += gui_menu.ecam_data.pos_offset.y;

                                vec3s brush_pnt = ecam.pos;
                                brush_pnt = glms_vec3_add(brush_pnt, glms_vec3_scale(ecam.right,brush_pnt_right));
                                brush_pnt = glms_vec3_add(brush_pnt, glms_vec3_scale(ecam.up,brush_pnt_up));

                                mat4s inv_trans;
                                glm_mat4_inv(editors[gui_menu.which_editor_selected].hmap_transform.matrix, inv_trans.raw);
                                // apply inverse transform
                                brush_pnt = glms_mat4_mulv3(inv_trans, brush_pnt, 1);

                                printf("%f %f\n", brush_pnt.x, brush_pnt.z);
                                break;

                        case EDITOR_ACTION_DELETE:
                                //only delete an editor we selected a valid editor
                                if (edit_idx != 0 && gui_menu.which_editor_selected < edit_idx) {
                                        editor_free_safe(editors, gui_menu.which_editor_selected, edit_idx);

                                        // shift every editor down if we delete one in the middle of the array
                                        for (int i = gui_menu.which_editor_selected; i < edit_idx; i++) {
                                                editors[i] = editors[i + 1];
                                        }

                                        edit_idx--;

                                        editor_free_safe(editors, edit_idx, edit_idx + 1);
                                }

                                gui_menu.editor_action = EDITOR_ACTION_IDLE;
                                break;
                        case EDITOR_ACTION_MOVE:
                                //TODO
                                break;
						case EDITOR_ACTION_BRUSH:
       						// Example: Right now just print the brush position.
      				    	// Replace this with your actual brush code later.
       						printf("Brush tool activated at %.3f, %.3f\n",
                			gui_menu.ecam_data.mouse_offset.x,
                			gui_menu.ecam_data.mouse_offset.y);
						    // Always reset the action after handling it
        					gui_menu.editor_action = EDITOR_ACTION_IDLE;
       						break;
					
                        default:
                        break;
                }
                
                
                glfwSwapBuffers(wnd);
                glfwPollEvents();
        }

        glfwTerminate();
        return 0;
}
