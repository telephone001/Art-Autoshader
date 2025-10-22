
#include "debug.h"






/// @brief helper function for a macro function that prints error 
/// @param filename the current file name you 
/// @param lineno line number
/// @param funcname function name
/// @param err error message
void print_err_(const char *filename, int lineno, const char *funcname, const char *err) 
{
        fprintf(stderr, "ERROR in file %s func %s line %d: %s\n", filename, funcname, lineno, err);
}


/// @brief helper function for a macro function that prints the current glError
/// @param filename current file
/// @param lineno line number
/// @param funcname function name
void gl_print_err_(const char *filename, int lineno, const char *const funcname)
{
        const char *err_msg = NULL;
	switch (glGetError()) {
	case GL_INVALID_ENUM: 			err_msg = "GL_INVALID_ENUM"; break;
	case GL_INVALID_VALUE: 			err_msg = "GL_INVALID_VALUE"; break;
	case GL_INVALID_OPERATION:		err_msg = "GL_INVALID_OPERATION"; break;
	case GL_STACK_OVERFLOW: 		err_msg = "GL_STACK_OVERFLOW"; break;
	case GL_STACK_UNDERFLOW:		err_msg = "GL_STACK_UNDERFLOW"; break;
	case GL_OUT_OF_MEMORY:			err_msg = "GL_OUT_OF_MEMORY"; break;
	case GL_INVALID_FRAMEBUFFER_OPERATION:	err_msg = "GL_INVALID_FRAMEBUFFER_OPERATION"; break;
	default: return;
	}

        print_err_(filename, lineno, funcname, err_msg);
}



