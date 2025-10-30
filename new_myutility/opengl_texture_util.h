#ifndef __MYUTIL_OPENGL_TEXTURE_UTIL__
#define __MYUTIL_OPENGL_TEXTURE_UTIL__

#include <general/debug.h>

#include <glad/glad.h>
#include "opengl_util.h"

#include <stb_image.h>

struct RenderData skybox_buffer_setup();

int load_2dtexture(GLuint *tex, char *const texture_path);

GLuint cubemap_create(const char *const *const tex_path_names, GLenum img_format);

#endif