#include "opengl_texture_util.h"



/// @brief GENERATES a texture object in tex.
///          if this function cannot make a texture, tex is unmodified and an error value is returned
/// @param tex the GL handler for the texture.
/// @param texture_path a string pointing to the path of the texture
/// @return negative values for error. 0 for success
int load_2dtexture(GLuint *tex, char *const texture_path)
{
	//needed so images aren't upside down
	stbi_set_flip_vertically_on_load(1);

	//check if texture path is valid
	int width, height, nr_channels; 
	unsigned char *data = stbi_load(texture_path, &width, &height, &nr_channels, 0);
	ERR_ASSERT_RET((data != NULL), -1, "texture could not be loaded");


	//the format of the image (determined by the channels)
	GLenum img_format = -1;
	switch (nr_channels) {
	case 1:
		img_format = GL_RED;
		break;	
	case 3:
		img_format = GL_RGB;
		break;
	case 4:
		img_format = GL_RGBA;
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


	glTexImage2D(GL_TEXTURE_2D, 0, 
		     img_format, width, height, 
		     0, img_format, GL_UNSIGNED_BYTE, data);
	
	glGenerateMipmap(GL_TEXTURE_2D);
	

	//This is important
	stbi_image_free(data);

	return 0;
}

