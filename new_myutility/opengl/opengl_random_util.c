#include "opengl_random_util.h"

#include <objloader/fast_obj.h>

struct RenderData skybox_buffer_setup()
{
	float vertices[] = {
	-1.0f, -1.0f, -1.0f,
	 1.0f, -1.0f, -1.0f,
	-1.0f,  1.0f, -1.0f,
	 1.0f,  1.0f, -1.0f,
	-1.0f, -1.0f,  1.0f,
	 1.0f, -1.0f,  1.0f,
	-1.0f,  1.0f,  1.0f,
	 1.0f,  1.0f,  1.0f,
	};

	unsigned int indices[] = {
	//front
	0, 1, 2,
	1, 2, 3,
	//back
	4, 5, 6,
	5, 6, 7,
	//left
	0, 2, 4,
	2, 4, 6,
	//right
	1, 3, 5,
	3, 5, 7,
	//top 
	2, 3, 6,
	3, 6, 7,
	//bottom
	0, 1, 4,
	1, 4, 5
	};

	struct RenderData skybox = renderdata_init_clear();
	skybox.vertices_length = ARRAY_LENGTH(vertices);
	skybox.indices_length = ARRAY_LENGTH(indices);
	skybox.vertices_stride = 3;
	skybox.indices_stride = 3;

	glGenVertexArrays(1, &skybox.vao);
	glGenBuffers(1, &skybox.vbo);
	glGenBuffers(1, &skybox.ebo);

	glBindVertexArray(skybox.vao);
	glBindBuffer(GL_ARRAY_BUFFER, skybox.vbo);

    	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, skybox.ebo);
    	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), &vertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
    	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, skybox.vertices_stride * sizeof(float), (void*)0);

	return skybox;
}

/// @brief creates a RenderData struct from an obj file using fastobj and mallocs fields
/// @param mdl_path name of the path of the model file
/// @param usage intended usage pattern of the data in buffer. Ex. GL_STATIC_DRAW
/// @return struct Renderdata has the vertices and the indices, the vao, vbo and the ebo
struct RenderData mdl_fastobj_buffer_setup_malloc(const char *const mdl_path, GLenum usage)
{
	struct RenderData model = renderdata_init_clear();
        
	fastObjMesh *mesh = fast_obj_read(mdl_path);
	if (!mesh) {
		fprintf(stderr, "Error: failed to locate or make mesh from obj file\n");
		return model;
	} 

	model.vertices_length = mesh->position_count * 3;
	
	model.vertices = malloc(model.vertices_length * sizeof(float));
	memcpy(model.vertices, mesh->positions, model.vertices_length * sizeof(float));

	model.indices_length = mesh->index_count;

	//modify indices to work with opengl
	model.indices = malloc(sizeof(unsigned int) * model.indices_length);
	for (int i = 0; i < model.indices_length; i++) {
                //extract the coordinates from the mesh struct data
		model.indices[i] = ((unsigned int*)mesh->indices)[i * 3];
	}

	//normal render setup
	bind_vao_and_vbo(
		&model.vao, 
		&model.vbo, 
		model.vertices,
		model.vertices_length * sizeof(float), 
		usage
	);

	bind_ebo(&model.ebo, model.indices, model.indices_length * sizeof(unsigned int), usage);

	//position attribute
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

 	fast_obj_destroy(mesh);
	return model;
}
