#include "opengl_texture_util.h"



/// @brief GENERATES a texture object in tex.
///          if this function cannot make a texture, tex is unmodified and an error value is returned
/// @param tex the GL handler for the texture.
/// @param texture_path a string pointing to the path of the texture
/// @param img_storage_format what format to store the image ex. GL_RGB GL_RGBA 
/// @return negative values for error. 0 for success
int load_2dtexture(GLuint *tex, char *const texture_path, GLint img_storage_format)
{
	//do not do stbi flip. The flipping will occur in the vertex shader or the vertex buffer

	//check if texture path is valid
	int width, height, nr_channels; 
	unsigned char *data = stbi_load(texture_path, &width, &height, &nr_channels, 0);
	ERR_ASSERT_RET((data != NULL), -1, "texture could not be loaded");


	//the format of the image (determined by the channels)
	GLenum img_input_format = -1;
	switch (nr_channels) {
	case 1:
		img_input_format = GL_RED;
		break;	
	case 3:
		img_input_format = GL_RGB;
		break;
	case 4:
		img_input_format = GL_RGBA;
		break;
	default:
		stbi_image_free(data);
		ERR_ASSERT_RET(0, -2, "image format could not be deduced");
	}
	

	//create a texture
	glGenTextures(1, tex); 
	glBindTexture(GL_TEXTURE_2D, *tex); 

	
	// set texture wrapping to GL_REPEAT (default wrapping method)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	//thank you olivarbo from learnopengl.com
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glTexImage2D(GL_TEXTURE_2D, 0, 
		     img_storage_format, width, height, 
		     0, img_input_format, GL_UNSIGNED_BYTE, data);
	
	glGenerateMipmap(GL_TEXTURE_2D);
	

	//This is important
	stbi_image_free(data);

	return 0;
}

