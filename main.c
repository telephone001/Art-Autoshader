#include <stdio.h>


#include "src/gui.h"
#include "general/debug.h"


#include "opengl_util.h"
#include "glfw_window.h"

#include <cglm/mat4.h>
#include <cglm/cam.h>
#include <cglm/util.h>
#include <cglm/struct.h>
#include <cglm/io.h>

#include <GLFW/glfw3.h>
#include <glad/glad.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_include.h>

#define SCR_LENGTH 800
#define SCR_HEIGHT 800

#define FOVY 45


extern Camera camera; //handler for cameradata;
extern int in_menu;   //menu status




void opengl_settings_init()
{
        //enable transparency
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);

        //sets up camera matrix
        uniform_buffer_setup();

        //static camera data
	mat4 cam_projection;
	glm_perspective(glm_rad(FOVY), (float)SCR_LENGTH / SCR_HEIGHT, 0.1f, 4000.0f, cam_projection);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(mat4), cam_projection);
        
        glClearColor(0.8,0.8,0.8,1);

}


void key_callback_menu_switching(GLFWwindow *wnd, int key, int scancode, int action, int mods)
{
        if (in_menu) {
                glfwSetScrollCallback(wnd, nk_gflw3_scroll_callback);
                glfwSetCharCallback(wnd, nk_glfw3_char_callback);
                glfwSetMouseButtonCallback(wnd, nk_glfw3_mouse_button_callback);

                //this will do the nuklear key callback
                nk_glfw3_key_callback(wnd, key, scancode, action, mods);

        } else {
	        glfwSetCursorPosCallback(wnd, mouse_callback);

                //this can be changed to the normal callbacks
                glfwSetScrollCallback(wnd, NULL);
                glfwSetCharCallback(wnd, NULL);
                glfwSetMouseButtonCallback(wnd, NULL);
	}


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



        GLuint plane = create_shader_program(
                "shaders/plane.vert",
                "shaders/plane.frag", 
                NULL, 
                NULL, 
                NULL
        );


        MenuOptions gui_menu;
        int err = nuklear_menu_init(&gui_menu, wnd, "fonts/american-typewriter.ttf", 22); 
        printf("%d\n", err);

        // this is required AFTER nuklear_menu_init because it uses the callbacks
        glfwSetKeyCallback(wnd, key_callback_menu_switching);
        

        while (!glfwWindowShouldClose(wnd)) {
                GL_PRINT_ERR();
                static float last_frame;
                float t = glfwGetTime();
		float delta_time = t - last_frame;
		last_frame = t;	
                

                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                //edit camera to look in correct direction
		mat4s cam_view = get_cam_view(camera);
		glBufferSubData(GL_UNIFORM_BUFFER, sizeof(mat4s), sizeof(mat4s), (float*)&cam_view);
		
		


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