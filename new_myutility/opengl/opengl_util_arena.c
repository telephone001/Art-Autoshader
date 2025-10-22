
#include "opengl_util_arena.h"

#define FAST_OBJ_IMPLEMENTATION
#include <objloader/fast_obj.h>


/// @brief creates a RenderData struct from an obj file using fastobj and arena allocs fields
/// @param mdl_path name of the path of the model file
/// @param usage intended usage pattern of the data in buffer. Ex. GL_STATIC_DRAW
struct RenderData mdl_fastobj_buffer_setup_arena(
	Arena *arena, 
	const char *const mdl_path, 
	GLenum usage
)
{
	struct RenderData model = {
		.vertices_stride = 3,
		.indices_stride = 3,

		//set below
		.vao = 0,
		.vbo = 0,
		.ebo = 0,
		.vertices_length = 0,
		.indices_length = 0,

		//allocated below
		.vertices = NULL,
		.indices = NULL,

		//NOT SET
		.polygon_mode = 0,
		.shader = 0,
	};
        
	fastObjMesh *mesh = fast_obj_read(mdl_path);
	if (!mesh) {
		fprintf(stderr, "Error: failed to locate or make mesh from obj file\n");
		return model;
	} 

	model.vertices_length = mesh->position_count * 3;
	
	model.vertices = arena_alloc(arena, model.vertices_length * sizeof(float));
	memcpy(model.vertices, mesh->positions, model.vertices_length * sizeof(float));

	model.indices_length = mesh->index_count;

	//modify indices to work with opengl
	model.indices = arena_alloc(arena, sizeof(unsigned int) * model.indices_length);
	for (int i = 0; i < model.indices_length; i++) {
                //extract the coordinates from the mesh struct data
		model.indices[i] = ((unsigned int*)mesh->indices)[i * 3];
	}

	//normal render setup
	bind_vao_and_vbo(&model.vao, &model.vbo, model.vertices, model.vertices_length * sizeof(float), GL_STATIC_DRAW);

	bind_ebo(&model.ebo, model.indices, model.indices_length * sizeof(unsigned int), GL_STATIC_DRAW);

	//position attribute
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

 	fast_obj_destroy(mesh);
	return model;
}

struct VertexBuffer mdl_fastobj_vertex_buffer_setup_arena(Arena *arena, const char *const mdl_path)
{
	fastObjMesh *mesh = fast_obj_read(mdl_path);

	if (!mesh) {
		fprintf(stderr, "Error: failed to locate or make mesh from obj file\n");
		return vertexbuffer_init_clear();
	} 

        VertexBuffer data = vertexbuffer_init_arena(arena, mesh->positions, mesh->position_count * 3, 3);
	
	return data;
}

/// @brief initializes MultBufferRenderData using arenas instead of malloc
///	  currently don't have a system to differentiate arena and malloced stuff so 
/// 	  might have to delete malloc mult_buffer.
/// @param num_buffers number of buffers to initialize
/// @param arena the arena you wanna allocate from
/// @param shader the shader that this renderdata use
/// @return MultBufferRenderData the mult buffer data
MultBufferRenderData mult_buff_renderdata_init_arena(
	const int num_buffers, 
	Arena *arena, 
	GLuint shader)
{
	
	MultBufferRenderData data = {
		//allocs an array of Vertex and Index buffers
		.vao = 0,
		.buffers = arena_alloc(arena, sizeof(VertexBuffer) * num_buffers),
		.buff_indices = arena_alloc(arena, sizeof(IndexBuffer) * num_buffers),
		.num_vertex_buffers = num_buffers,
		.polygon_mode = GL_FILL,
		.shader = shader,
	};

	glGenVertexArrays(1, &data.vao);
	
	return data;
}


/// @brief Creates a VBO, allocates a buffer to be stored inside the VertexBuffer, and buffers that
///	   data to the VBO. Also updates the other values of the VertexBuffer
/// @param arena the arena to allocate from
/// @param data the data that you want to put in the VertexBuffer
/// @param length length of the data you want to put in the VertexBuffer
/// @param stride the stride of the data that is put in the VertexBuffer
/// @return the VertexBuffer
VertexBuffer vertexbuffer_init_arena(
	Arena *arena, 
	const float *const data, 
	int length, 
	int stride)
{
	//invalid input means return empty vertex buffer
	if (length <= 0 || stride <= 0) {
		return vertexbuffer_init_clear();
	}

	size_t buffer_size = length * sizeof(float);

	VertexBuffer tmp = {
		.buff = arena_alloc(arena, buffer_size),
		.buff_len = length,
		.stride = stride,
		.vbo = 0
	};

	glGenBuffers(1, &tmp.vbo);
	
	//if there is data, copy it to the buff in vertexbuffer
	if (data) {
		memcpy(tmp.buff, data, buffer_size);
	}

	//send data to buffer
	glBindBuffer(GL_ARRAY_BUFFER, tmp.vbo);
	glBufferData(GL_ARRAY_BUFFER, buffer_size, data, GL_STATIC_DRAW);

	return tmp;
}


IndexBuffer indexbuffer_init_arena(
	Arena *arena, 
	const unsigned int *const data, 
	int length, 
	int stride)
{
	//invalid input means return empty vertex buffer
	if (length <= 0 || stride <= 0) {
		return indexbuffer_init_clear();
	}

	size_t buffer_size = length * sizeof(float);

	IndexBuffer tmp = {
		.indices = arena_alloc(arena, buffer_size),
		.indices_len = length,
		.stride = stride,
		.ebo = 0
	};

	glGenBuffers(1, &tmp.ebo);
	
	//if there is data, copy it to the buff in vertexbuffer
	if (data) {
		memcpy(tmp.indices, data, length * buffer_size);
	}

	//send data to buffer
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tmp.ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, buffer_size, data, GL_STATIC_DRAW);

	return tmp;
}



