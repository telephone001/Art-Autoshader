#ifndef __MYUTIL_OPENGL_TEXTURE_UTIL__
#define __MYUTIL_OPENGL_TEXTURE_UTIL__

#include <glad/glad.h>
#include "opengl_util.h"

struct RenderData skybox_buffer_setup();

GLuint load_2dtexture(const char *const texture_path);

GLuint cubemap_create(const char *const *const tex_path_names, GLenum img_format);

#endif