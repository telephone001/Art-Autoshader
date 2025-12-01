//******************************************************************************
// glfw_window.h
// Author: jchoi338@wisc.edu
//******************************************************************************
#ifndef __MYUTIL_GLFW_WINDOW_H__
#define __MYUTIL_GLFW_WINDOW_H__

#include <stdio.h>
#include <stdlib.h>

#include <glad/glad.h>

#include <cglm/cam.h>
#include <cglm/struct.h>

#include <GLFW/glfw3.h>

#define MOUSE_SENSITIVITY   	0.15f
#define NORMAL_CAMERA_SPEED 	10.0f

// camera parameters
#define FOVY 45
#define NEAR 0.1
#define FAR 4000

typedef struct Camera {
	vec3s pos;	
	vec3s front;	//normalized vector for the direction the camera is looking at
	vec3s up;	//normalized vector
	vec3s right;	//normalized vector
	float speed;
} Camera;


/**
 * @brief calculates the cam view from camera
 * 	  used when multiple things need the cam_view
 * 
 * @param camera camera struct
 * @return mat4s the cam view
 */
void get_cam_view(Camera camera, mat4 cam_view);

void mouse_callback(GLFWwindow *window, double x_pos, double y_pos);

void handle_wasd_move(GLFWwindow *const window, float delta_time);

GLFWwindow *glfw_setup(int major_version, 
		       int minor_version, 
		       unsigned int screen_width, 
		       unsigned int screen_height, 
		       const char *const window_name);

#endif