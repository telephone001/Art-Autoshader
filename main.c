#include <stdio.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_include.h>

#include "src/gui.h"
#include "src/editor.h"
#include "src/light_sources.h"

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

extern Camera camera; //handler for cameradata;
extern int in_menu;   //menu status


int debug_thing = 0; // TODO  REMOVE IN FINAL PRODUCT

/// @brief 
void opengl_settings_init()
{

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);

        //sets up camera matrix
        uniform_buffer_setup();

        //static camera data
	mat4 cam_projection;
	glm_perspective(glm_rad(FOVY), (float)SCR_LENGTH / SCR_HEIGHT, NEAR, FAR, cam_projection);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(mat4), cam_projection);
        
        //set background color
        glClearColor(0.4,0.4,0.4,1);

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

                if (key == GLFW_KEY_P && action == GLFW_PRESS) {
                        debug_thing = 1;
                }

                if (key == GLFW_KEY_L && action == GLFW_PRESS) {
                        debug_thing = 2;
                }

                if (key == GLFW_KEY_1 && action == GLFW_PRESS) {
                        debug_thing = 3;
                }

                if (key == GLFW_KEY_2 && action == GLFW_PRESS) {
                        debug_thing = 4;
                }
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



int main() 
{
        if (!glfwInit()) {
                return -1; // Initialization failed
        }

        GLFWwindow* wnd = glfw_setup(4, 1, SCR_LENGTH, SCR_HEIGHT, "art autoshader");
	    if (!wnd)
	    	return -1;

        opengl_settings_init();


        MenuOptions gui_menu;
        int err = nuklear_menu_init(&gui_menu, wnd, "fonts/american-typewriter.ttf", 22); 
        ERR_ASSERT_RET((err >= 0), -1, "nuklear window could not be created");

        // this is required AFTER nuklear_menu_init because it uses the callbacks
        glfwSetKeyCallback(wnd, key_callback_menu_switching);



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
		mat4s cam_view = get_cam_view(camera);
		glBufferSubData(GL_UNIFORM_BUFFER, sizeof(mat4s), sizeof(mat4s), (float*)&cam_view);
		

                // If debug thing is 1, we make an editor
                if (debug_thing == 1) {

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
                        debug_thing = 0;
                        cnt++;

                        if (cnt >= MAX_EDITORS) {
                                cnt = 0;
                        }
                }

                // delete editor
                if (debug_thing == 2) {
                        for (int i = 0; i < MAX_EDITORS; i++) {
                                //Required before editor_free if gui_menu is using the texture.
                                if (editors[i].mdl_cam_plane.textures != NULL && 
                                    gui_menu.img_tex == editors[i].mdl_cam_plane.textures[0]) {
                                        editors[i].mdl_cam_plane.textures[0] = 0;
                                }
                                editor_free(&(editors[i]));
                        }

                        debug_thing = 0;
                }

                if (debug_thing == 3) {
                        //TODO check this error
                        light_source_add(&light_sources_data, (LightSource){POINT, camera.pos});
                        debug_thing = 0;
                }

                if (debug_thing == 4) {
                        light_source_add(&light_sources_data, (LightSource){DIRECTIONAL, camera.pos});
                        debug_thing = 0;
                }
                
                //Render the editors
                glEnable(GL_DEPTH_TEST);
                glDepthFunc(GL_LESS);
                for (int i = 0; i < MAX_EDITORS; i++) {
                        if (editors[i].mdl_cam_proj.vao != 0 ) {
                                editor_render(&(editors[i]));
                        }
                }
                
                // REMEMBER THAT THIS CAN RETURN AN ERROR
                light_sources_render(&light_sources_data);


		if (in_menu) {
                        nuklear_menu_render(wnd, &gui_menu);
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