
#ifndef __MYUTIL_OPENGL_RBO_UTIL_H__
#define __MYUTIL_OPENGL_RBO_UTIL_H__

#include <glad/glad.h>
#include <GLFW/glfw3.h>


/// @brief creates a color texture for the fbo to use (does NOT make mipmaps)
/// @param fbo the frame buffer object
/// @param width the width of the texture you want to allocate
/// @param height the height of the texture you want to allocate
/// @return the texture corresponding to the fbo
GLuint fbo_tex_init(GLuint fbo, int width, int height);


/// @brief creates an rbo for an fbo
/// @param fbo the frame buffer object that you want to create an rbo for
/// @return renderbuffer object
GLuint fbo_rbo_init(GLuint fbo, int width, int height);








#endif
