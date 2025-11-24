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


/// @brief function that calculates the width and height of an image that you want to fit inside a bounded box
///		with width bound_w and height bound_h. Returns values through r_width and r_height
/// @param r_width returned width of image
/// @param r_height returned height of image
/// @param aspect_ratio aspect ratio of image
/// @param bound_w width bound on the image
/// @param bound_h height bound on the image
/// @return 
int img_rect_fit(float *r_width, float *r_height, float aspect_ratio, float bound_w, float bound_h)
{
	if (!r_width || !r_height)
		return -1;

	//invalid inputs
	if (aspect_ratio <= 0 || bound_w <= 0 || bound_h <= 0)
		return -2;

	//the img will always be bounded by two faces if it truly fits the bound

	//try setting height to be bound_h.
	if (bound_h * aspect_ratio <= bound_w) {
		//width fits
		*r_width = bound_h * aspect_ratio;
		*r_height = bound_h;
	} else {
		*r_width = bound_w;
		*r_height = bound_w / aspect_ratio;
	}

	return 0;
}

/// @brief gets the aspect ratio from the texture width/height WARNING: binds the texture
/// @param tex texture
/// @return the aspect ratio
float img_aspect_ratio(GLuint tex)
{
        glBindTexture(GL_TEXTURE_2D, tex);
	int w, h;
	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);

        return (float)w/(float)h;
}