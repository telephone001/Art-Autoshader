#include <stdio.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_include.h>

#include "src/gui.h"
#include "src/editor.h"
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

// camera parameters
#define FOVY 45
#define NEAR 0.1
#define FAR 4000

#define CAM_PROJ_MDL_DIST 20

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
/// @param rdata the outputted render data for the wireframe of a camera projection
/// @param shader the shader is a parameter because I don't want to create a new shader object for each model
/// @param fovy the fovy of the camera.
/// @param aspect_ratio the aspect ratio of the camera width/height.
/// @param cam the camera object that you want to make a model of
/// @param plane_dist the distance between the camera and the plane looked at
/// @return 0 on success negatives on error
int cam_proj_mdl_init(
        RenderData *rdata, 
        GLuint shader,
        float aspect_ratio, 
        float fovy, 
        Camera cam,
        float plane_dist
)
{

        vec3s plane_pos = glms_vec3_add(cam.pos, glms_vec3_scale(cam.front, plane_dist));


        //the dist between rectangle origin and rectangle top
        float dist_up = sin(fovy/2) * glms_vec3_norm(glms_vec3_sub(plane_pos, cam.pos));

        //the dist between rectangle origin and the rightost point of the rectangle
        float dist_right = dist_up * aspect_ratio;

        //find the four rectangle points:
        //vec3s r0 = plane_pos - cam_up * dist_up - cam.right * dist_right
        //vec3s r1 = plane_pos - cam_up * dist_up + cam.right * dist_right
        //vec3s r2 = plane_pos + cam_up * dist_up - cam.right * dist_right
        //vec3s r3 = plane_pos + cam_up * dist_up + cam.right * dist_right

        vec3s rect[4];
        for (int i = 0; i < 4; i++) {
                rect[i] = plane_pos;
        }
        
        rect[0] = glms_vec3_add(rect[0], glms_vec3_scale(cam.up, -dist_up));
        rect[1] = glms_vec3_add(rect[1], glms_vec3_scale(cam.up, -dist_up));
        rect[2] = glms_vec3_add(rect[2], glms_vec3_scale(cam.up,  dist_up));
        rect[3] = glms_vec3_add(rect[3], glms_vec3_scale(cam.up,  dist_up));

        rect[0] = glms_vec3_add(rect[0], glms_vec3_scale(cam.right, -dist_right));
        rect[1] = glms_vec3_add(rect[1], glms_vec3_scale(cam.right,  dist_right));
        rect[2] = glms_vec3_add(rect[2], glms_vec3_scale(cam.right, -dist_right));
        rect[3] = glms_vec3_add(rect[3], glms_vec3_scale(cam.right,  dist_right));



        //we should have 5 points for our vertices. (the 4 above and the cam pos) 
        //Now we start filling in the renderdata

        *rdata = (RenderData) {
	        
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
	        .shader = shader,
        };

        ERR_ASSERT_RET((rdata->vertices != NULL), -2, "malloc failed");
        ERR_ASSERT_RET((rdata->indices != NULL), -2, "malloc failed");

        //copy all points into renderdata. the first vec3 will be the cam pos
        memcpy(rdata->vertices, cam.pos.raw, sizeof(vec3s));
        memcpy(rdata->vertices + 3, (float*)rect, sizeof(vec3s) * 4);

        //make a line for each possible connection in the index buffer
        {
                int cnt = 0; //
                for (unsigned int i = 0; i < 5; i++) {
                        for (unsigned int j = i+1; j < 5; j++) {
                                rdata->indices[2 * cnt] = i;
                                rdata->indices[2 * cnt + 1] = j;
                                cnt++;
                        }
                }
        }


        

        bind_vao_and_vbo(&(rdata->vao), &(rdata->vbo), rdata->vertices, sizeof(float) * rdata->vertices_length, GL_STATIC_DRAW);
        
        ERR_ASSERT_RET((rdata->vao != 0), -3, "vao failed");
        ERR_ASSERT_RET((rdata->vbo != 0), -4, "vbo failed");


        bind_ebo(&(rdata->ebo), rdata->indices, sizeof(unsigned int) * rdata->indices_length, GL_STATIC_DRAW);

        ERR_ASSERT_RET((rdata->vao != 0), -5, "ebo failed");

        
	//position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, rdata->vertices_stride * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
        

        return 0;
}

/// @brief initializes a cam plane model (plane that includes the image but not the heightmap)
///            This is used inside the editor model to display the image on the plane
/// @param cam_plane the struct we output to hold the cam plane
/// @param cam_proj the struct we get the cam plane from. This is a cam_proj_mdl
/// @param shader shader for the plane containing a texture
/// @param img_tex the texture of the image
/// @return 
static int cam_plane_mdl_init(RenderData *cam_plane, RenderData *cam_proj, GLuint shader, GLuint img_tex) 
{
        *cam_plane = (RenderData) {
	        
                // will be filled out below
                .vao = 0, 
	        .vbo = 0, 
	        .ebo = 0, 

	        .vertices = malloc((sizeof(vec3) + sizeof(vec2)) * 4), //we only need 4 points. All from cam_mdl
	        .vertices_stride = 5,                  //has to be 5 cuz of vec3 for pos and 2 for texcord
	        .vertices_length = 5 * 4,

	        .indices = malloc(sizeof(unsigned int) * 3 * 2), // 3 for each of the 2 triangles
	        .indices_stride = 3, // 3 for triangle primitives
	        .indices_length = 3 * 2,

	        .textures = malloc(sizeof(GLuint)), //you gotta alloc memory so that the menu image isn't the same as the projected image 
	        .num_textures = 1,   

	        .primitive_type = GL_LINES,
	        .shader = shader,
        };



        //copy all plane points from cam_proj to the cam_plane vertices
        for (int i = 0; i < cam_plane->vertices_length; i++) {
                memcpy(
                        cam_plane->vertices + i * cam_plane->vertices_stride,
                        cam_proj->vertices + 3 + (3 * i),
                        sizeof(vec3s)
                );
        }

        //initialize texture coords for vertices
        {
                float tex_coords[] = {
                        0,1,
                        1,1,
                        0,0,
                        1,0,
                }; 
                for (int i = 0; i < cam_plane->vertices_length; i++) {
                        memcpy(
                                cam_plane->vertices + i * cam_plane->vertices_stride + 3,
                                tex_coords + 2*i,
                                sizeof(vec2s)
                        );
                }
        }


        //copy indices to renderdata
        {
                unsigned int indices[] = {
                        0,1,2,
                        0,2,3,
                };

                memcpy(
                        cam_plane->indices,
                        indices,
                        sizeof(indices)
                );
        }

        *cam_plane->textures = img_tex;


        //now do regular opengl initializaiton
        bind_vao_and_vbo(&(cam_plane->vao), &(cam_plane->vbo), cam_plane->vertices, sizeof(float) * cam_plane->vertices_length, GL_STATIC_DRAW);
        
        ERR_ASSERT_RET((cam_plane->vao != 0), -3, "vao failed");
        ERR_ASSERT_RET((cam_plane->vbo != 0), -4, "vbo failed");


        bind_ebo(&(cam_plane->ebo), cam_plane->indices, sizeof(unsigned int) * cam_plane->indices_length, GL_STATIC_DRAW);

        ERR_ASSERT_RET((cam_plane->vao != 0), -5, "ebo failed");

        
	//position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, cam_plane->vertices_stride * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

        // texture attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, cam_plane->vertices_stride * sizeof(float), (void*)(sizeof(vec2s)));
	glEnableVertexAttribArray(1);

        return 0;
}

/// @brief renders a cam_plane model
/// @param cam_plane_rdata the renderdata of the plane you want to render.
void cam_plane_mdl_render(RenderData *cam_plane_rdata)
{
        glBindVertexArray(cam_plane_rdata->vao);
        glBindTexture(GL_TEXTURE_2D, cam_plane_rdata->textures[0]);

	glUseProgram(cam_plane_rdata->shader);
	glDrawElements(cam_plane_rdata->primitive_type, cam_plane_rdata->indices_length, GL_UNSIGNED_INT, 0);
}

/// @brief simple command to render a cam_proj model. 
/// @param cam_proj_rdata the renderdata of the thing you want to render (has to be initialized first)
void cam_proj_mdl_render(RenderData *cam_proj_rdata)
{
        glBindVertexArray(cam_proj_rdata->vao);
	glUseProgram(cam_proj_rdata->shader);
	glDrawElements(cam_proj_rdata->primitive_type, cam_proj_rdata->indices_length, GL_UNSIGNED_INT, 0);
}

int editor_mdl_init(Editor *editor, GLFWwindow *wnd, MenuOptions *gui_menu, GLuint shader_cam_proj, GLuint shader_cam_plane)
{

        int width, height;
        glfwGetFramebufferSize(wnd, &width, &height);

        int err;

        err = cam_proj_mdl_init(
                &editor->mdl_cam_plane,
                shader_cam_proj,
                (float)width/(float)height,
                FOVY,
                camera,
                CAM_PROJ_MDL_DIST  
        );

        ERR_ASSERT_RET((err == 0), -1, "cam_proj_mdl_init failed");

        err = cam_plane_mdl_init(
                &editor->mdl_cam_plane,
                &editor->mdl_cam_proj,
                shader_cam_plane,
                gui_menu->img_tex
        );

        ERR_ASSERT_RET((err == 0), -2, "cam_plane_mdl_init failed");


        return 0;
}

int editor_render(Editor *editor)
{
        return 0;
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



        GLuint shader_basic = create_shader_program(
                "shaders/basic.vert",
                "shaders/basic.frag", 
                NULL, 
                NULL, 
                NULL
        );

	ERR_ASSERT_RET((shader_basic != 0), -2, "basic shader didn't work");

        RenderData cam_proj_mdl[100] = {0};
        int cnt = 0;

        glfwSetInputMode(wnd, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        
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
		


                if (debug_thing == 1) {
                        render_data_free(&(cam_proj_mdl[cnt]));
                        int width, height;
                        glfwGetFramebufferSize(wnd, &width, &height);

                        cam_proj_mdl_init(
                                &(cam_proj_mdl[cnt]), 
                                shader_basic,
                                (float)width / (float)height, 
                                FOVY, 
                                camera, 
                                CAM_PROJ_MDL_DIST
                        );
                        debug_thing = 0;
                        cnt++;

                        if (cnt >= 100) {
                                cnt = 0;
                        }
                }

                for (int i = 0; i < 100; i++) {
                        if (cam_proj_mdl[i].vao) {
                                cam_proj_mdl_render(&(cam_proj_mdl[i]));
                        }
                }


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