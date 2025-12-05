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


#define SCR_LENGTH 800
#define SCR_HEIGHT 800

#define MAX_EDITORS 100


#define ORTHO_SCALE 1.0 / 100.0

extern Camera camera; //handler for cameradata;
extern int in_menu;   //menu status


typedef enum DebugThing {
        DEBUG_NONE,
        SPAWN_EDITOR,
        DELETE_EVERYTHING,
        SPAWN_POINT_LIGHT,
        SPAWN_DIRECTIONAL_LIGHT,
        SPAWN_LIGHTS_FOR_CAMERA_RAYS,
        SPAWN_LIGHTS_FOR_INTERSECTIONS,
} DebugThing;

DebugThing debug_thing = 0; // TODO  REMOVE IN FINAL PRODUCT

//TODO these are only in global because I dont want to send a window hint pointer to callback

mat4 flycam_projection;
mat4 flycam_view;

mat4 offset_view;
mat4 ortho_proj; //used to actually edit the heightmap



/// @brief 
void opengl_settings_init()
{

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

                if (key == GLFW_KEY_P && action == GLFW_PRESS) 
                        debug_thing = SPAWN_EDITOR;
                

                if (key == GLFW_KEY_L && action == GLFW_PRESS) 
                        debug_thing = DELETE_EVERYTHING;
                

                if (key == GLFW_KEY_1 && action == GLFW_PRESS) 
                        debug_thing = SPAWN_POINT_LIGHT;
                

                if (key == GLFW_KEY_2 && action == GLFW_PRESS) 
                        debug_thing = SPAWN_DIRECTIONAL_LIGHT;
                
                if (key == GLFW_KEY_5 && action == GLFW_PRESS) 
                        debug_thing = SPAWN_LIGHTS_FOR_CAMERA_RAYS;

                if (key == GLFW_KEY_6 && action == GLFW_PRESS)
                        debug_thing = SPAWN_LIGHTS_FOR_INTERSECTIONS;
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
        if (!glfwInit()) {
                return -1; // Initialization failed
        }

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
        ERR_ASSERT_RET((err >= 0), -1, "nuklear window could not be created");

        // this is required AFTER nuklear_menu_init because it uses the callbacks
        glfwSetKeyCallback(wnd, key_callback_menu_switching);


        LightSourcesData light_sources_data = {0};

        GLuint point_light_tex, directional_light_tex;
        load_2dtexture(&point_light_tex, "textures/point_light.jpg", GL_RGB);
        load_2dtexture(&directional_light_tex, "textures/directional_light.jpg", GL_RGB);

        err = light_sources_rd_init(&(light_sources_data.rd), shader_light_sources, point_light_tex, directional_light_tex);
        ERR_ASSERT_RET((err >= 0), -2, "light sources struct could not be created");


        // THIS IS FOR DEBUGGING
        Editor editors[MAX_EDITORS] = {0};
        int cnt = 0;

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

                        // WARNING:  EDITOR DOESNT DO REFERENCE COUNTING FOR ALLOCATED TEXTURES
                        //Required before editor_free if gui_menu is using the texture.
                        if (editors[cnt].mdl_cam_plane.textures != NULL &&
                            gui_menu.img_tex == editors[cnt].mdl_cam_plane.textures[0]) {
                                editors[cnt].mdl_cam_plane.textures[0] = 0;
                        }
                        editor_free(&(editors[cnt]));
                        int err = editor_init(
                                &(editors[cnt]), 
                                wnd, 
                                &gui_menu, 
                                shader_cam_proj, 
                                shader_cam_plane,
                                shader_hmap,
                                100, // PLACHOLDER
                                100  // PLACHOLDER
                        );
                        
                        hmap_edit_sinc(&(editors[cnt]));


                        if (err < 0) {
                                //Required before editor_free if gui_menu is using the texture.
                                if (editors[cnt].mdl_cam_plane.textures != NULL &&
                                    gui_menu.img_tex == editors[cnt].mdl_cam_plane.textures[0]) {
                                        editors[cnt].mdl_cam_plane.textures[0] = 0;
                                }
                                editor_free(&(editors[cnt]));
                        }
                        debug_thing = DEBUG_NONE;
                        cnt++;

                        if (cnt >= MAX_EDITORS) {
                                cnt = 0;
                        }
                }

                // delete editor AND light sources
                if (debug_thing == DELETE_EVERYTHING) {
                        for (int i = 0; i < MAX_EDITORS; i++) {
                                //Required before editor_free if gui_menu is using the texture.
                                if (editors[i].mdl_cam_plane.textures != NULL && 
                                    gui_menu.img_tex == editors[i].mdl_cam_plane.textures[0]) {
                                        editors[i].mdl_cam_plane.textures[0] = 0;
                                }
                                editor_free(&(editors[i]));
                        }

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
                
                // RAYTRACING TIME !  ! !
                if (debug_thing == SPAWN_LIGHTS_FOR_CAMERA_RAYS) {
                        int width, height;
	                glfwGetWindowSize(wnd, &width, &height);
                        //array of rays
                        vec3s *cam_dirs = ht_generate_camera_directions(&camera, width, height);


                        for (int i = 0; i < width; i++) {
                                for (int j = 0; j < height; j++) {
                                        if (((j + 10) % 20 == 0 && (i + 10) % 20 == 0)) {
                                                light_source_add(&light_sources_data, (LightSource){DIRECTIONAL, glms_vec3_add(camera.pos, glms_vec3_scale(cam_dirs[i*width + j],5)), 0.05});
                                        }
                                }
                        }
                        //for (int i = 0; i < height; i++) {
                        //        for (int j = 0; j < width; j++) {
                        //                //how far the ray goes
                        //                float t_ray;
                        //                vec3s point;
                        //                int shot = ht_intersect_heightmap_ray(
                        //                    editors[0].hmap, editors[0].hmap_w, editors[0].hmap_l, camera.pos, cam_dirs[i*width + j],
                        //                    0.1, 10, &t_ray, &point);
                        //                
                        //                        
                        //                if (shot != 0) {
                        //                        printf("SHOT");
                        //                        vec3s point2 = camera.pos;
                        //                        vec3s tmp = glms_vec3_scale(cam_dirs[i * height + j], t_ray);
                        //                        point2 = glms_vec3_add(camera.pos, tmp);
                        //                        light_source_add(&light_sources_data, (LightSource){POINT, point2, 0.05});
                        //                }
                        //        }
                        //}

                        free(cam_dirs);

                        debug_thing = DEBUG_NONE;
                }

                if (debug_thing == SPAWN_LIGHTS_FOR_INTERSECTIONS) {
                        int width, height;
	                glfwGetWindowSize(wnd, &width, &height);
                        //array of rays
                        vec3s *cam_dirs = ht_generate_camera_directions(&camera, width, height);


                        for (int i = 0; i < height; i++) {
                                for (int j = 0; j < width; j++) {
                                        if (j % 50 != 0 || i % 50 != 0) {
                                                continue;
                                        }

                                        //how far the ray goes
                                        float t_ray;
                                        vec3s point;
                                        int shot = ht_intersect_heightmap_ray(
                                            editors[0].hmap, editors[0].hmap_w, editors[0].hmap_l, camera.pos, cam_dirs[i*width + j],
                                            0.1, 500, &t_ray, &point);
                                        
                                                
                                        if (shot != 0) {
                                                printf("SHOT\n");
                                                vec3s point2 = camera.pos;
                                                vec3s tmp = glms_vec3_scale(cam_dirs[i * height + j], t_ray);
                                                point2 = glms_vec3_add(camera.pos, tmp);
                                                light_source_add(&light_sources_data, (LightSource){POINT, point2, 1});
                                        }
                                }
                        }

                        free(cam_dirs);

                        debug_thing = DEBUG_NONE;
                }


                //Render the editors
                glEnable(GL_DEPTH_TEST);
                glDepthFunc(GL_LESS);
                
                for (int i = 0; i < MAX_EDITORS; i++) {
                        if (editors[i].mdl_cam_proj.vao != 0 ) {
                                editor_render(&(editors[i]), 0, flycam_projection, flycam_view);
                        }
                }
                
///////////////////////////////// TESTS ON FRAMEBUFFER ////////////////////////////////////
                GLint window_w, window_h;
                glfwGetWindowSize(wnd, &window_w, &window_h); // Get main window size for restore
                glBindFramebuffer(GL_FRAMEBUFFER, gui_menu.ecam_data.fbo);

                
                // *** CRITICAL FIX: Set Viewport to Framebuffer/Texture Size ***
                // Assuming gui_menu.ecam_data holds the dimensions of the framebuffer texture.
                // Replace these placeholders with the correct fields if necessary.
                glViewport(0, 0, gui_menu.ecam_data.width, gui_menu.ecam_data.height); 
 
                // Use the standard clear color (dark blue) to confirm if the clear worked.
                glClearColor(0.1, 0.2, 0.3, 1.0);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                Camera editor_cam = camera;
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
                                        (gui_menu.ecam_data.in_perspective) ? flycam_projection : ortho_proj, 
                                        offset_view
                                );
                        }
                }

                glBindFramebuffer(GL_FRAMEBUFFER, 0);

                // *** CRITICAL FIX: Restore Viewport to Main Window Size ***
                glViewport(0, 0, window_w, window_h);

///////////////////////////////// TESTS ON FRAMEBUFFER ////////////////////////////////////

                
                // REMEMBER THAT THIS CAN RETURN AN ERROR
                light_sources_render(&light_sources_data, flycam_projection, flycam_view);


		if (in_menu) {
                        nuklear_menu_render(wnd, delta_time, &gui_menu);
        		glfwSetCursorPosCallback(wnd, NULL);
        	} else {
                	handle_wasd_move(wnd, delta_time);
                	glfwSetCursorPosCallback(wnd, mouse_callback);
		}
                
                glfwSwapBuffers(wnd);
                glfwPollEvents();
        }

        glfwTerminate();
        return 0;
}