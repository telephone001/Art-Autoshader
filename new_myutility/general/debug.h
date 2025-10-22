#ifndef __MYUTIL_DEBUG_H__
#define __MYUTIL_DEBUG_H__

#include <GLAD/glad.h>
#include <stdio.h>


#define PRINT_ERR(err) print_err_(__FILE__, __LINE__, __FUNCTION__, err)
#define GL_PRINT_ERR() gl_print_err_(__FILE__, __LINE__, __FUNCTION__)




// will assert something. If it fails, it will return an int 
// THIS WILL ONLY PRINT IF ERR_ASSERT_RET_PRINT IS DEFINED
#define ERR_ASSERT_RET(cond, ret, msg)                                                                      \
do {                                                                                                        \
        if (!cond) {                                                                                        \
#ifdef ERR_ASSERT_RET_PRINT                                                                                 \
                fprintf(stderr, "ERROR in file %s func %s line %d: %s\n", filename, funcname, lineno, msg); \
#endif                                                                                                      \
        return ret;                                                                                         \
        }                                                                                                   \
} while (0)




void gl_print_err_(const char *filename, int lineno, const char *const funcname);
void print_err_(const char *filename, int lineno, const char *funcname, const char *err); 

#endif