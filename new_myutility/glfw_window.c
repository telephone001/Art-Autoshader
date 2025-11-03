//******************************************************************************
// glfw_window.c
// Author: jchoi338@wisc.edu
//******************************************************************************

#include "glfw_window.h"

//globals
Camera camera = {
	.pos = 		{0.0f, 0.0f, 0.0f},
	.front = 	{0.0f, 0.0f, -1.0f},
	.up = 		{0.0f, 1.0f, 0.0f},
	.right = 	{1.0f, 0.0f, 0.0f},
	.speed = 	NORMAL_CAMERA_SPEED
};

int in_menu = 1;

/**
 * @brief sets up a uniform buffer for the camera defined above
 * 	  used when you want to use the ubo for camera variables
 * 	  Currently consists of a camera view matrix and a projection matrix
 *
 */
GLuint uniform_buffer_setup()
{
	GLuint ubo;
	glGenBuffers(1, &ubo);
	glBindBuffer(GL_UNIFORM_BUFFER, ubo);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, ubo, 0, 2 * sizeof(mat4));
	glBufferData(GL_UNIFORM_BUFFER, 2 * sizeof(mat4), NULL, GL_STATIC_DRAW);
}

/**
 * @brief calculates the cam view from camera
 * 	  used when multiple things need the cam_view
 * 
 * @param camera camera struct
 * @return mat4s the cam view
 */
mat4s get_cam_view(Camera camera)
{
	mat4s cam_view;
	vec3s view_center;
	view_center = glms_vec3_add(camera.pos, camera.front);
	return glms_lookat(camera.pos, view_center, camera.up);
}

/**
 * @brief buffers the camera view to the camera uniform buffer provided in the glfw_window file.
 * 	  used for shortening the code for sending view to the ubo.
 */
void ubo_cam_view_edit()
{
	mat4s cam_view = get_cam_view(camera);
        glBufferSubData(GL_UNIFORM_BUFFER, sizeof(mat4s), sizeof(mat4s), &cam_view);
}

/**
 * @brief handles the wasd movement for a glfw window 
 * 	MODIFIES THE CAMERA GLOBAL VARIABLE
 * 
 * @param window the window handler
 * @param delta_time change in time from last glfwGetTime
 */
void handle_wasd_move(GLFWwindow *const window, float delta_time)
{
	vec3s front_dir;
	
	front_dir = glms_vec3_scale(camera.front, camera.speed * delta_time);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		camera.pos = glms_vec3_add(camera.pos, front_dir);
	}
	glfwPollEvents();
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		camera.pos = glms_vec3_sub(camera.pos, front_dir);
	}
	
	//if a or d, change move direction to left or right
	vec3s right_dir  = glms_vec3_scale(camera.right, camera.speed * delta_time);

	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		camera.pos = glms_vec3_sub(camera.pos, right_dir);
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		camera.pos = glms_vec3_add(camera.pos, right_dir);
	}
}

void mouse_callback(GLFWwindow *window, double x_pos, double y_pos)
{
	static int first_mouse = 1; //checks if this is the first time the function was called

	static float last_x;
	static float last_y;

	//initialize a yaw and pitch
	static float yaw = -90.0f;
	static float pitch = 0.0f;

	//if this is the first time the function is run, use current mouse pos as last mouse pos
	if (first_mouse) {
		last_x = x_pos;
		last_y = y_pos;
		first_mouse = 0;
	}

	//find the offset of the mouse position
	float offset_x = x_pos - last_x;
	float offset_y = last_y - y_pos; //reversed because y coordinates go from bottom to top

	//after using the last_y and last_x, replace them with the current x_pos and y_pos for next time.
	last_x = x_pos;
	last_y = y_pos;

	//update the jaw and pitch
	yaw += offset_x * MOUSE_SENSITIVITY;
	pitch += offset_y * MOUSE_SENSITIVITY;

	//when the pitch is out of bounds, this prevents the screen from rolling over
	if (pitch > 89.9f)
		pitch = 89.9f;
	if (pitch < -89.9f)
		pitch = -89.9f;

	//change the camera front value
	camera.front = (vec3s){
		.x = cos(glm_rad(yaw)) * cos(glm_rad(pitch)),
		.y = sin(glm_rad(pitch)),
		.z = sin(glm_rad(yaw)) * cos(glm_rad(pitch))
	};

	//updates the camera's up value so movement works when looking down/up
	camera.up = glms_vec3_cross(camera.front, (vec3s){0.0f, 1.0f, 0.0f});
	camera.up = glms_vec3_normalize(camera.up);
	camera.up = glms_vec3_cross(camera.up, camera.front);
	camera.up = glms_vec3_normalize(camera.up);

	camera.right  = glms_normalize(glms_cross(camera.front, camera.up));
}


/// @brief will make sure the viewport matches the new dimensions when resized and also buffers the projection matrix to the uniform buffer
/// @param window window of glfw program
/// @param width width of screen
/// @param height height of screen
void framebuffer_size_callback(GLFWwindow *const window, int width, int height)
{
	//change the projection matrix of the camera uniform buffer
	mat4s cam_projection = glms_perspective(glm_rad(45.0f), (double)width / (double)height, 0.1f, 4000.0f);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(mat4), (float*)&cam_projection);

	//resize the viewport
	glViewport(0, 0, width, height);
}

GLFWwindow *glfw_setup(int major_version, int minor_version, 
		       unsigned int screen_width, unsigned int screen_height, 
		       const char *const window_name)
{
	if (!glfwInit()) 
		return NULL;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major_version);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor_version);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	GLFWwindow *window = glfwCreateWindow(screen_width, screen_height, window_name, NULL, NULL);

	if (!window) {
		fprintf(stderr, "failed to initialize GLFW\n");
		glfwTerminate();
		return NULL;
	}

	glfwMakeContextCurrent(window);
	gladLoadGL();
	glViewport(0, 0, screen_width, screen_height);


	//callbacks
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);


	

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	return window;
}



