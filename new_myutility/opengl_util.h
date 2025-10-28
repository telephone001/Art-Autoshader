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

	GLenum primitive_type;
	GLuint shader;
} RenderData;

RenderData renderdata_init_clear();

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