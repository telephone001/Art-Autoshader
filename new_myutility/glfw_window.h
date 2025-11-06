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

typedef struct Camera {
	vec3s pos;	
	vec3s front;	//normalized vector for the direction the camera is looking at
	vec3s up;	//normalized vector
	vec3s right;	//normalized vector
	float speed;
} Camera;

// this is the layout for the uniform buffer you can use
/*
layout (std140) uniform cameraData {
        mat4 projection;
        mat4 view;
};    
*/

GLuint uniform_buffer_setup();

mat4s get_cam_view(Camera camera);

void ubo_cam_view_edit();

void framebuffer_size_callback(GLFWwindow *const window, int width, int height);

void mouse_callback(GLFWwindow *window, double x_pos, double y_pos);

void handle_wasd_move(GLFWwindow *const window, float delta_time);

GLFWwindow *glfw_setup(int major_version, 
		       int minor_version, 
		       unsigned int screen_width, 
		       unsigned int screen_height, 
		       const char *const window_name);

#endif