
#ifndef __MYUTIL_OPENGL_UTIL_ARENA_H__
#define __MYUTIL_OPENGL_UTIL_ARENA_H__

#include "general/arena.h"
#include "opengl_util.h"

struct RenderData mdl_fastobj_buffer_setup_arena(
	Arena *arena, 
	const char *const mdl_path, 
	GLenum usage
);

struct VertexBuffer mdl_fastobj_vertex_buffer_setup_arena(Arena *arena, const char *const mdl_path);

MultBufferRenderData mult_buff_renderdata_init_arena(
	const int num_buffers, 
	Arena *arena, 
	GLuint shader
);

VertexBuffer vertexbuffer_init_arena(
	Arena *arena, 
	const float *const data, 
	int length, 
	int stride
);

IndexBuffer indexbuffer_init_arena(
	Arena *arena, 
	const unsigned int *const data, 
	int length, 
	int stride
);

#endif