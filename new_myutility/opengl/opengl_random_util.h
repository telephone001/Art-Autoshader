#ifndef __MYUTIL_OPENGL_RANDOM_UTIL__
#define __MYUTIL_OPENGL_RANDOM_UTIL__

#include "general/arena.h"

#include "opengl_util.h"
#include "opengl_util_arena.h"

//#include "objloader/fast_obj.h"

#include <glad/glad.h>

struct RenderData skybox_buffer_setup();

struct RenderData mdl_fastobj_buffer_setup_malloc(const char *const mdl_path, GLenum usage);

#endif
