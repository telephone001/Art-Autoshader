
#include "opengl_rbo_util.h"


/// @brief creates a color texture for the fbo to use (does NOT make mipmaps)
/// @param fbo the frame buffer object
/// @param width the width of the texture you want to allocate
/// @param height the height of the texture you want to allocate
/// @return the texture corresponding to the fbo
GLuint fbo_tex_init(GLuint fbo, int width, int height)
{
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        GLuint tex;

        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(
                GL_TEXTURE_2D, 
                0, 
                GL_RGBA8, 
                width, 
                height, 
                0, 
                GL_RGBA, 
                GL_UNSIGNED_BYTE,
                NULL
        );

        // do NOT use a mipmap. I believe the renderbuffer will be changed every frame.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);

        return tex;
}


/// @brief creates an rbo for an fbo
/// @param fbo the frame buffer object that you want to create an rbo for
/// @return renderbuffer object
GLuint fbo_rbo_init(GLuint fbo, int width, int height)
{
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        GLuint rbo;

        glGenRenderbuffers(1, &rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo);

        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height); 
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

        return rbo;
}

