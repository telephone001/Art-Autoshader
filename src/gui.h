#ifndef __GUI_H__
#define __GUI_H__

#include "opengl_util.h"
#include "general/debug.h"

#include <GLFW/glfw3.h>

#define  MAX_VERTEX_BUFFER 512 * 1024
#define  MAX_ELEMENT_BUFFER 128 * 1024
#define  NK_INCLUDE_FIXED_TYPES
#define  NK_INCLUDE_STANDARD_IO
#define  NK_INCLUDE_STANDARD_VARARGS
#define  NK_INCLUDE_DEFAULT_ALLOCATOR
#define  NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define  NK_INCLUDE_FONT_BAKING
#define  NK_INCLUDE_DEFAULT_FONT
#define  NK_KEYSTATE_BASED_INPUT

#include <nuklear/nuklear.h>
#include <nuklear/nuklear_glfw_gl3.h>

#include <STB/stb_include.h>

#define GUI_NK_MAX_INPUT_LEN 17


#define GUI_IMG_PATH_BUFF_LEN 255 // the buffer length used to store the path of the image

// If we ever want to make more states, we can. But right now there will only be one state
typedef enum MenuState {
        MENU_STATE_MAIN
} MenuState;


/// @brief 
typedef struct MenuOptions {
        struct nk_context *ctx; //context for nuklear window
        MenuState state;        //what window the program is in

        int font_size; //will govern the size of the letters and textboxes in the gui

        char img_path[GUI_IMG_PATH_BUFF_LEN]; //path where the image is stored
        GLuint img_tex;   //glfw texture id of the image we want to use
        struct nk_image img_nk; //nk handler of the image

} MenuOptions;





/// @brief will initialize settings of nuklear gui and will give you a gui menu object
/// @param gui_menu menu object created by this function
/// @param wnd glfw window
/// @return error code. negative values for errors
int nuklear_menu_init(MenuOptions *gui_menu, GLFWwindow *wnd);



/// @brief renders the nuklear menu. Use this in the drawloop
/// @param wnd the window of the program to render the gui on
void nuklear_menu_render(GLFWwindow *wnd);


#endif