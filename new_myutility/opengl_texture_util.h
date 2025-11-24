#ifndef __MYUTIL_OPENGL_TEXTURE_UTIL__
#define __MYUTIL_OPENGL_TEXTURE_UTIL__

#include <general/debug.h>

#include <glad/glad.h>
#include "opengl_util.h"

#include <stb_image.h>

int load_2dtexture(GLuint *tex, char *const texture_path, GLint img_storage_format);

/// @brief function that calculates the width and height of an image that you want to fit inside a bounded box
///		with width bound_w and height bound_h. Returns values through r_width and r_height
/// @param r_width returned width of image
/// @param r_height returned height of image
/// @param aspect_ratio aspect ratio of image
/// @param bound_w width bound on the image
/// @param bound_h height bound on the image
/// @return 
int img_rect_fit(float *r_width, float *r_height, float aspect_ratio, float bound_w, float bound_h);

/// @brief gets the aspect ratio from the texture width/height WARNING: binds the texture
/// @param tex texture
/// @return the aspect ratio
float img_aspect_ratio(GLuint tex);

#endif