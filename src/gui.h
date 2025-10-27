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

#include <nuklear.h>
#include <nuklear_glfw_gl3.h>

#include <stb_include.h>

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

        char img_path[GUI_IMG_PATH_BUFF_LEN]; //buffer where the image is stored. (textbox buffer)
        GLuint img_tex;   //glfw texture id of the image we want to use
        struct nk_image img_nk; //nk handler of the image

} MenuOptions;





/// @brief This will do three main things:
///		call nk_glfw3_init(wnd, NK_GLFW3_INSTALL_CALLBACKS) to set up the gui
///		fill in the struct containing menu data
/// 		set up the font of the gui using "fonts/american-typewriter.ttf"
///
/// @param gui_menu output: filled out gui_menu struct
/// @param wnd glfw window handler
/// @param font_path the path to the font you want to choose for the gui
/// @param font_size how big the letters and textboxes should be in the gui
/// @return 
int nuklear_menu_init(MenuOptions *gui_menu, GLFWwindow *wnd, const char *const font_path, int font_size);





void nuklear_menu_render(GLFWwindow *wnd, MenuOptions *const gui_menu);


#endif