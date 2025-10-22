#ifndef __MYUTIL_DEBUG_H__
#define __MYUTIL_DEBUG_H__

#include <GLAD/glad.h>
#include <stdio.h>


#define ERR_ASSERT_RET_PRINT // for debugging FOR NOW




#define PRINT_ERR(err) print_err_(__FILE__, __LINE__, __FUNCTION__, err)
#define GL_PRINT_ERR() gl_print_err_(__FILE__, __LINE__, __FUNCTION__)


// will assert something. If it fails, it will return an int 
// THIS WILL ONLY PRINT IF ERR_ASSERT_RET_PRINT IS DEFINED
#ifdef ERR_ASSERT_RET_PRINT
        #define ERR_ASSERT_RET(cond, ret, msg)                                                                             \
        do {                                                                                                               \
                if (!cond) {                                                                                               \
                        fprintf(stderr, "ERROR in file %s func %s line %d: %s\n", __FILE__,  __FUNCTION__, __LINE__, msg); \
                        return ret;                                                                                        \
                }                                                                                                          \
        } while (0)
#else 
        #define ERR_ASSERT_RET(cond, ret, msg)                                                                             \
        do {                                                                                                               \
                if (!cond) {                                                                                               \
                        return ret;                                                                                        \
                }                                                                                                          \
        } while (0)
#endif       






void gl_print_err_(const char *filename, int lineno, const char *const funcname);
void print_err_(const char *filename, int lineno, const char *funcname, const char *err); 

#endif