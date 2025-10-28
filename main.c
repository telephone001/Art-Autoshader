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
void key_callback_menu_switching(GLFWwindow *wnd, int key, int scancode, int action, int mods)
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


/// @brief will output the renderdata of a wireframe of a camera projection. Note that this renderdata allocates 
///             memory for its vertex buffer! make sure to free it!!!
/// @param render_data the outputted render data for the wireframe of a camera projection
/// @param fovy the fovy of the camera.
/// @param aspect_ratio the aspect ratio of the camera.
/// @param cam_pos the 3d position of the camera
/// @param plane_pos the 3d position of the center of a perpendicular plane of the camera
/// @param up the camera's up vector. Has to be normalized
/// @return 0 on success negatives on error
int cam_proj_mdl_init(
        RenderData *render_data, 
        float aspect_ratio, 
        float fovy, 
        vec3s cam_pos, 
        vec3s plane_pos, 
        vec3s cam_up
)
{
        //calculate the 3d positions of the corners of the plane the camera looks at (NOT NORMALIZED)
        vec3s cam_front = glms_vec3_sub(plane_pos, cam_pos);

        //nomalized vectors of right and forward
        vec3s right_dir = glms_vec3_crossn(cam_front, cam_up);

        //the dist between rectangle origin and rectangle top
        float dist_up = sin(fovy/2) * glms_vec3_norm(cam_front);

        //the dist between rectangle origin and the rightost point of the rectangle
        float dist_right = dist_up * aspect_ratio;

        //find the four rectangle points:
        //vec3s r0 = cam_pos - cam_up * dist_up - right_dir * dist_right
        //vec3s r1 = cam_pos - cam_up * dist_up + right_dir * dist_right
        //vec3s r2 = cam_pos + cam_up * dist_up - right_dir * dist_right
        //vec3s r3 = cam_pos + cam_up * dist_up + right_dir * dist_right

        vec3s rect[4];
        for (int i = 0; i < 4; i++) {
                rect[i] = cam_pos;
        }
        
        rect[0] = glms_vec3_add(rect[0], glms_vec3_scale(cam_up, -dist_up));
        rect[1] = glms_vec3_add(rect[1], glms_vec3_scale(cam_up, -dist_up));
        rect[2] = glms_vec3_add(rect[2], glms_vec3_scale(cam_up,  dist_up));
        rect[3] = glms_vec3_add(rect[3], glms_vec3_scale(cam_up,  dist_up));

        rect[0] = glms_vec3_add(rect[0], glms_vec3_scale(right_dir, -dist_right));
        rect[1] = glms_vec3_add(rect[1], glms_vec3_scale(right_dir,  dist_right));
        rect[2] = glms_vec3_add(rect[2], glms_vec3_scale(right_dir, -dist_right));
        rect[3] = glms_vec3_add(rect[3], glms_vec3_scale(right_dir,  dist_right));



        //we should have 5 points for our vertices. (the 4 above and the cam pos) 
        //Now we start filling in the renderdata

        GLuint shader_basic = create_shader_program(
                "shaders/basic.vert",
                "shaders/basic.frag", 
                NULL, 
                NULL, 
                NULL
        );
	ERR_ASSERT_RET((shader_basic != 0), -1, "basic shader didn't work");

        *render_data = (RenderData) {
	        
                // will be filled out below
                .vao = 0, 
	        .vbo = 0, 
	        .ebo = 0, 

	        .vertices = malloc(sizeof(vec3s) * 5), //we only need 5 points
	        .vertices_stride = 3,                  //has to be 3 cuz of vec3
	        .vertices_length = 3 * 5,

	        .indices = malloc(sizeof(unsigned int) * 2 * 10), //10 because there are 10 ways to connect each of the 5 points
	        .indices_stride = 2, // 2 because we are rendering lines
	        .indices_length = 2 * 10,

	        .textures = NULL, //no textures
	        .num_textures = 0,    // nope.

	        .primitive_type = GL_LINES,
	        .shader = shader_basic,
        };

        ERR_ASSERT_RET((render_data->vertices != NULL), -2, "malloc failed");
        ERR_ASSERT_RET((render_data->indices != NULL), -2, "malloc failed");
        

        bind_vao_and_vbo(&(render_data->vao), &(render_data->vbo), render_data->vertices, sizeof(float) * render_data->vertices_length, GL_STATIC_DRAW);
        
        ERR_ASSERT_RET((render_data->vao != 0), -3, "vao failed");
        ERR_ASSERT_RET((render_data->vbo != 0), -4, "vbo failed");

        
        bind_ebo(&(render_data->ebo), render_data->indices, sizeof(unsigned int) * render_data->indices_length, GL_STATIC_DRAW);

        ERR_ASSERT_RET((render_data->vao != 0), -5, "ebo failed");

        
	//position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, render_data->vertices_stride * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
        

        return 0;
}

/// @brief simple command to render a cam_proj model. 
/// @param cam_proj_render_data the renderdata of the thing you want to render (has to be initialized first)
void cam_proj_mdl_render(RenderData *cam_proj_render_data)
{
        glBindVertexArray(cam_proj_render_data->vao);
	glUseProgram(cam_proj_render_data->shader);
	glDrawElements(cam_proj_render_data->primitive_type, cam_proj_render_data->indices_length, GL_UNSIGNED_INT, 0);
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