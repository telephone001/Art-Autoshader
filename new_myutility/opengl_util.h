//******************************************************************************
// opengl_util.h
//
//******************************************************************************
#ifndef __MYUTIL_OPENGL_UTIL_H__
#define __MYUTIL_OPENGL_UTIL_H__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <glad/glad.h>

//remove when not debugging
#define MYUTIL_DEBUG

#define MAX_SHADER_BUF_LEN	0x10000000

//only works for stack arrays
#define ARRAY_LENGTH(x) (sizeof(x) / sizeof((x)[0]))

//standard renderdata needed for rendering.
typedef struct RenderData {
	GLuint vao;
	GLuint vbo;
	GLuint ebo;

	float *vertices;
	unsigned int vertices_stride;
	unsigned int vertices_length;

	unsigned int *indices;
	unsigned int indices_stride;
	unsigned int indices_length;

	GLuint *textures;
	int num_textures;

	GLenum polygon_mode;
	GLuint shader;
} RenderData;

/**
 * @brief a single vertex buffer. 
 * 
 */
typedef struct VertexBuffer {
	GLuint vbo;
	float *data;
	unsigned int length;
	unsigned int stride;
} VertexBuffer;

typedef struct IndexBuffer {
	GLuint ebo;
	unsigned int *data;
	unsigned int length;
	unsigned int stride;
} IndexBuffer;

//if there are no indices for the buffer at position, the indices should be empty
typedef struct MultBufferRenderData {
	GLuint vao;
	//the vertex buffers
	VertexBuffer *buffers;

	//the index buffers. each index of buff_indices should correspond to a buffer.
	//if a buffer doesn't have an index buffer attached to it at an index, the item 
	//at that index should be null
	IndexBuffer *buff_indices;

	//number of buffers. this should be the length of both arrays
	int num_vertex_buffers;

	GLuint *textures;
	int num_textures;

	GLenum polygon_mode;

	GLuint shader;
} MultBufferRenderData;

RenderData renderdata_init_clear();

VertexBuffer vertexbuffer_init_clear();

IndexBuffer indexbuffer_init_clear();

MultBufferRenderData mult_buff_renderdata_init_malloc(const int num_buffers, const int num_textures, GLuint shader);


void mult_buff_renderdata_vertexbuffer_set(
        int attrib_num, 
        int stride, 
        int buffer_length,
        float *buffer, 
	size_t buffer_offset,
        GLenum draw_type,
        MultBufferRenderData *renderdata
);

void mult_buff_renderdata_indexbuffer_set(
        int attrib_num,
        int stride,
        int index_buffer_length,
        unsigned int *indices, 
        MultBufferRenderData *renderdata
);

void mult_buff_renderdata_malloc_free(MultBufferRenderData data);

void print_error(FILE *error_stream, int line_number, const char *const function_name);

GLuint compile_shader(const char *const path, int type);

GLuint add_shader(GLuint shader_program, const char *const path, unsigned int shader_type);

GLuint create_shader_program(const char *const vertexPath, 
			     const char *const fragPath, 
			     const char *const geomPath,
			     const char *const tessControlPath,
			     const char *const tessEvalPath);

#ifdef GL_COMPUTE_SHADER
GLuint compute_shader_create(const char *const compute_shader_path);
#endif

void bind_vao_and_vbo(GLuint *vao, GLuint *vbo, float *vertices, size_t vertices_size, GLenum usage);

void bind_ebo(GLuint *ebo, unsigned int *indices, size_t indices_size, GLenum usage);

#endif // __MYUTIL_OPENGL_UTIL_H__