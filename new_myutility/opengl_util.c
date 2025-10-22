//******************************************************************************
// opengl_util.c
// define MYUTIL_DEBUG for debug messages
//******************************************************************************

#include "opengl_util.h"

/// @brief initializes a renderdata to default fields
/// 
/// @return RenderData of renderdata
RenderData renderdata_init_clear()
{
	RenderData data = {
	.vao             = 0,
	.vbo             = 0,
	.ebo             = 0,
	.vertices 	 = NULL,
	.vertices_stride = 0,
	.vertices_length = 0,
	.indices 	 = NULL,
	.indices_stride  = 0,
	.indices_length  = 0,
	.polygon_mode 	 = GL_FILL,
	.shader 	 = 0,
	.textures	 = NULL,
	};
	
	return data;
}


/// @brief returns a vertexbuffer that is effectively 0
/// 
/// @return VertexBuffer vertexbuffer with null pointer and all arguments 0
VertexBuffer vertexbuffer_init_clear()
{
	return (VertexBuffer) {
		.data = NULL,
		.length = 0,
		.stride = 0,
		.vbo = 0
	};
}

/// @brief returns an indexbuffer that is effectively 0
/// 
/// @return IndexBuffer with null pointer and all other arguments 0
IndexBuffer indexbuffer_init_clear()
{
	return (IndexBuffer) {
		.data = NULL,
		.length = 0,
		.stride = 0,
		.ebo = 0
	};
}

/// @brief initializes  mult buffer renderdata using malloc and also creates a vao
/// 
/// @param num_buffers number of Vertex and Indices buffers to be malloced. Buffers are not 
/// 		      initialized and are freshly malloced
/// 			
/// @param shader shader that this renderdata uses
/// @return MultBufferRenderData 
MultBufferRenderData mult_buff_renderdata_init_malloc(const int num_buffers, const int num_textures, GLuint shader)
{
	MultBufferRenderData data = {
		.buffers = malloc(sizeof(VertexBuffer) * num_buffers),
		.buff_indices = malloc(sizeof(IndexBuffer) * num_buffers),
		.num_vertex_buffers = num_buffers,
		.polygon_mode = GL_FILL,
		.textures = malloc(sizeof(GLuint) * num_textures),
		.num_textures = num_textures,
		.shader = shader
	};

	glGenVertexArrays(1, &data.vao);

	return data;
}

/// @brief frees a mult buffer renderdata that was initialized using malloc. Shader is not deleted
/// 
/// @param data the renderdata to destroy
void mult_buff_renderdata_malloc_free(MultBufferRenderData data) 
{
	if (data.buffers != NULL) {
		for (int i = 0; i < data.num_vertex_buffers; i++) {
			if (data.buffers[i].data != NULL) {
				free(data.buffers[i].data);
			}
		}
		free(data.buffers);
	}
	
	if (data.buff_indices != NULL) {
		for (int i = 0; i < data.num_vertex_buffers; i++) {
			if (data.buff_indices[i].data != NULL) {
				free(data.buff_indices[i].data);
			}
		}
		free(data.buff_indices);
	}

	free(data.textures);
	glDeleteVertexArrays(1, &data.vao);
}

/// @brief fills out the buffer of a multbufferrenderdata given an element attribute and 
///          all of its other information. Will malloc a buffer corresponding to the element attribute
/// @param attrib_num  type of attribute you want to fill out in the renderdata
/// @param stride       stride of the buffer
/// @param buffer_length length of the buffer you want to make
/// @param buffer copies this buffer's stuff into the renderdata vertexbuffer's pointer.
/// @param buffer_offset BYTE offset from beginning of buffer stride row to get the data from
///                  if null, the renderdata vertex buffer is not filled but still allocated
/// @param draw_type type of opengl draw (GL_STATIC_DRAW, GL_STREAM_DRAW, GL_DYNAMIC_DRAW)
/// @param renderdata renderdata to modify the contents of vertex buffer
void mult_buff_renderdata_vertexbuffer_set(
        int attrib_num, 
        int stride, 
        int buffer_length,
        float *buffer, 
	size_t buffer_offset,
        GLenum draw_type,
        MultBufferRenderData *renderdata)
{
        size_t buffer_size = sizeof(float) * buffer_length;
        renderdata->buffers[attrib_num].data = malloc(buffer_size);

        if (buffer)
                memcpy(renderdata->buffers[attrib_num].data, buffer, buffer_size);

        //generate vbo & bind it to the vertex buffer associated with the attribute number
        glGenBuffers(1, &renderdata->buffers[attrib_num].vbo);
        glBindBuffer(GL_ARRAY_BUFFER, renderdata->buffers[attrib_num].vbo);

        renderdata->buffers[attrib_num].length = buffer_length;
        renderdata->buffers[attrib_num].stride = stride;

        glVertexAttribPointer(
                attrib_num, 
                stride,
                GL_FLOAT, 
                GL_FALSE, 
                stride * sizeof(float), 
                (void*)(buffer_offset)
        );
	glEnableVertexAttribArray(attrib_num);

        if (buffer)
	        glBufferData(GL_ARRAY_BUFFER, buffer_size, buffer, draw_type);
}

/// @brief fills out the indexbuffer of a multbufferrenderdata given an element attribute and 
///          all of its other information. Will malloc an index buffer corresponding to the element attribute
/// @param attrib_num type of attribute you want to fill out in the renderdata
/// @param stride       stride of the index buffer
/// @param index_buffer_length length of the index buffer
/// @param indices      pointer to indices you want to copy to the indexbuffer holder, if null,
///                     the buffer is not filled but is still allocated
/// @param renderdata   the renderdata you want to modify
void mult_buff_renderdata_indexbuffer_set(
        int attrib_num,
        int stride,
        int index_buffer_length,
	unsigned int *indices, 
        MultBufferRenderData *renderdata
)
{
        size_t index_buffer_size = index_buffer_length * sizeof(unsigned int);

        //fill out indexbuffer data
        glGenBuffers(1, &renderdata->buff_indices[attrib_num].ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, renderdata->buff_indices[attrib_num].ebo);

        renderdata->buff_indices[attrib_num].data = malloc(index_buffer_size);
        if (indices)
                memcpy(renderdata->buff_indices[attrib_num].data, indices, index_buffer_size);

        renderdata->buff_indices[attrib_num].length = index_buffer_length;
        renderdata->buff_indices[attrib_num].stride = stride;

        glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_buffer_size, indices, GL_STATIC_DRAW);
}


/// @brief reads shader files and stores result in a string that is returned. 
/// 		  Prone to bugs because of limited buffer and uses malloc
/// @param filename name of the file that is read
/// @return char* the string containing the text contents of the file.
/// 			returns NULL when there is an error reading the file
static char *retrieve_shader_file(const char *const file_name)
{
	char *shader_string = malloc(MAX_SHADER_BUF_LEN + 1);
	FILE *shader_file;

	//open a file and do error checking
	shader_file = fopen(file_name, "r");
	if (shader_file == NULL) {
		printf("error: %s shader file could not be found\n", file_name);
		return NULL;
	}
	
	//read file and store how many characters it has read
	size_t shader_size = fread(shader_string, 
				    sizeof(char), 
				    MAX_SHADER_BUF_LEN, 
				    shader_file);
	
	if (!feof(shader_file))
		fprintf(stderr, "error: %s too big to be put in shader."
				 "Increase macro MAX_SHADER_BUF_LEN\n", file_name);

	if (ferror(shader_file) != 0) {
		printf("error reading shader file\n");
		fclose(shader_file);
		return NULL;
	}

	shader_string[shader_size] = '\0';
	shader_size++;

	shader_string = realloc(shader_string, shader_size);

	fclose(shader_file);
	return shader_string;
}

/// @brief compiles the shader using its file path and type
/// 
/// @param path shader file path as a string
/// @param type type of shader to compile as a macro
/// @return GLuint the shader id
GLuint compile_shader(const char *const path, int type)
{
	char *shader_source = retrieve_shader_file(path);
	if (!shader_source) 
		return 0;
	
	GLuint shader_id = glCreateShader(type);
	glShaderSource(shader_id, 1, (char const *const *)&shader_source, NULL);
	glCompileShader(shader_id);

#ifdef MYUTIL_DEBUG
	GLint success = 0;
	glGetShaderiv(shader_id, GL_COMPILE_STATUS, &success);
	if (success == GL_FALSE) {
		free(shader_source);
		return 0;
	}
		
#endif
	free(shader_source);

	return shader_id;
}

 
/// @brief compiles and attaches the shader to a shader program
/// 
/// @param shader_program id of the shader program
/// @param path shader file path as a string
/// @param shader_type type of shader to compile as a macro
/// @return GLuint the shader id
GLuint add_shader(GLuint shader_program, const char *const path, unsigned int shader_type)
{
	if (!path) 
		return 0;

	GLuint shader = compile_shader(path, shader_type);
	glAttachShader(shader_program, shader);
	glDeleteShader(shader);

	return shader;
}

/// @brief makes the vertex and fragment shaders for opengl to use
/// 
/// @param vertex_path path to vertex shader file as a string.
/// @param frag_path path to frag shader file as a string.
/// @param geom_path path to geometry shader file as a string. NULL if no geometry shader
/// @param tess_control_path path to tesselation control shader file as a string. NULL if no tesselation control shader
/// @param tess_eval_path path to tesselation evaluation shader file as a string. NULL if no tesselation evaluation shader
/// 
/// @return shader handler. returns 0 on error
GLuint create_shader_program(const char *const vertex_path, 
			     const char *const frag_path, 
			     const char *const geom_path,
			     const char *const tess_control_path,
			     const char *const tess_eval_path)
{
	GLuint shader_program = glCreateProgram();

	if (!add_shader(shader_program, vertex_path, GL_VERTEX_SHADER)) {
		fprintf(stderr, "Failed to add %s\n", vertex_path);
		return 0;
	}
	
	if (!add_shader(shader_program, frag_path, GL_FRAGMENT_SHADER)) {
		fprintf(stderr, "Failed to add %s\n", frag_path);
		return 0;
	}

	//optional shaders
#ifdef GL_GEOMETRY_SHADER
	if (!add_shader(shader_program, geom_path, GL_GEOMETRY_SHADER) && 
	    geom_path != NULL) {
		fprintf(stderr, "Failed to add %s\n", geom_path);
		return 0;
	}
#endif

#ifdef GL_TESS_CONTROL_SHADER
	if (!add_shader(shader_program, tess_control_path, GL_TESS_CONTROL_SHADER) &&
	    tess_control_path != NULL) {
		fprintf(stderr, "Failed to add %s\n", tess_control_path);
		return 0;
	}
#endif

#ifdef GL_TESS_EVALUATION_SHADER
	if (!add_shader(shader_program, tess_eval_path, GL_TESS_EVALUATION_SHADER) &&
	    tess_eval_path != NULL) {
		fprintf(stderr, "Failed to add %s\n", tess_eval_path);
		return 0;
	}
#endif

	//link the shaders together
	glLinkProgram(shader_program);

	//check for linking error
	GLint success;
	glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
    	if (!success) {
    	    	GLchar info_log[512];
    	    	glGetProgramInfoLog(shader_program, 512, NULL, info_log);
    	    	printf("Linking error: %s\n", info_log);
    	}
 
	return shader_program;
}


#ifdef GL_COMPUTE_SHADER
GLuint compute_shader_create(const char *const compute_shader_path)
{
	GLuint program = glCreateProgram();
	if (!add_shader(program, compute_shader_path, GL_COMPUTE_SHADER)) {
		fprintf(stderr, "Failed to add %s\n", compute_shader_path);
		return 0;
	}
	glLinkProgram(program);

	return program;
}
#endif

/// @brief generates & binds the vao and vbo for you using a vertices array
/// 
/// @param vao pointer to vao
/// @param vbo pointer to vbo
/// @param vertices pointer to vertices
/// @param vertices_size size of vertices
/// @param usage usage pattern of data storage. ex. GL_STATIC_DRAW
void bind_vao_and_vbo(GLuint *vao, GLuint *vbo, float *vertices, size_t vertices_size, GLenum usage)
{
	glGenVertexArrays(1, vao);
	glGenBuffers(1, vbo);

	glBindVertexArray(*vao);

	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, vertices_size, vertices, usage);
}

/// @brief generates & binds the vao and vbo for you using a vertices array
/// 
/// @param ebo pointer to ebo
/// @param indices pointer to indices
/// @param indices_size size of indices
/// @param usage usage pattern of data storage. ex. GL_STATIC_DRAW
void bind_ebo(GLuint *ebo, unsigned int *indices, size_t indices_size, GLenum usage)
{
	glGenBuffers(1, ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *ebo);
    	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_size, indices, usage);
}


