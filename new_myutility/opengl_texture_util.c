#include "opengl_texture_util.h"


#include <STB/stb_include.h>


/**
 * @brief creates a cubemap texture ID from 6 inputted textures
 * 
 * @param tex_path_names string array of 6 textures you want in the cubemap
 * @param img_format format of image. ex. GL_RGB, GL_RGBA
 * @return GLuint cubemap texture id
 */
GLuint cubemap_create(const char *const *const tex_path_names, GLenum img_format)
{
	GLuint texture;

	int width, height, nr_channels; 
	glGenTextures(1, &texture); 

	glBindTexture(GL_TEXTURE_CUBE_MAP, texture); 

	for (int i = 0; i < 6; i++) {
		unsigned char *data = stbi_load(tex_path_names[i], 
						&width, &height, 
						&nr_channels, 0);

		if (data) {
    			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 
				     0, img_format, width, height, 
				     0, img_format, GL_UNSIGNED_BYTE, data);
    			glGenerateMipmap(GL_TEXTURE_2D);
    		} else {
			printf("cubemap failed with texture %s\n", tex_path_names[i]);
			return 0;
		}

		stbi_image_free(data);
	}

	// set texture wrapping to GL_REPEAT
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE); 	

    	// set texture filtering parameters
    	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	return texture;
}

static int stbi_get_data_type(const char *const img_path, GLenum *img_format) 
{
	//TODO needs to be tested
	GLenum fmt;
	
	int width, height, nrchannels;
	stbi_info(img_path, &width, &height, &nrchannels);

	switch (nrchannels) {
	case 1:
		fmt = GL_RED;
		break;	
	case 3:
		fmt = GL_RGB;
		break;
	case 4:
		fmt = GL_RGBA;
		break;
	default:
		return GL_RGB;
	}
	
	
	//modify the return pointer
	*img_format = fmt;
	return 0;
}

/// @brief returns a texture handler given a texture path
/// @param texture_path the texture path
/// @return the texture handler
GLuint load_2dtexture(const char *const texture_path)
{
	//needed so images aren't upside down
	stbi_set_flip_vertically_on_load(1);

	GLuint texture;
	
	int width, height, nr_channels; 
	glGenTextures(1, &texture); 
	unsigned char *data = stbi_load(texture_path, &width, &height, &nr_channels, 0);

	glBindTexture(GL_TEXTURE_2D, texture); 

	GLenum img_format;
	//get the image data type
	if (stbi_get_data_type(texture_path, &img_format)) {
		printf("texture img_format not recognized in \"%s\" :(\n", texture_path);
	}

	
	// set texture wrapping to GL_REPEAT (default wrapping method)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


	if (data) {
	    	glTexImage2D(GL_TEXTURE_2D, 0, 
			     img_format, width, height, 
			     0, img_format, GL_UNSIGNED_BYTE, data);
		
	    	glGenerateMipmap(GL_TEXTURE_2D);
	} else {
		printf("failed to make texture \"%s\" :(\n", texture_path);
	}

	stbi_image_free(data);

	return texture;
}

