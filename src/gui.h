#ifndef __GUI_H__
#define __GUI_H__

#include "cglm/struct.h"
#include "opengl_util.h"
#include "opengl_texture_util.h"
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

#define GUI_IMG_FILENAME_MAX_LEN 255 //the max length of a filename Does not include null terminator

#define GUI_IMG_PATH_START "input_image/"        //the path relative to main.c where the code will look for the input files
#define GUI_IMG_PATH_START_LEN sizeof(GUI_IMG_PATH_START) / sizeof(GUI_IMG_PATH_START[0])    //the size of the path start. Includes null terminator
#define GUI_IMG_PATH_START_IDX GUI_IMG_PATH_START_LEN - 1 //start modifying at the null terminator

// the buffer length used to store the path of the image
// note that the gui_img_path_buff_len includes the null terminator in its length.
#define GUI_IMG_PATH_BUFF_LEN   GUI_IMG_PATH_START_LEN + GUI_IMG_FILENAME_MAX_LEN 



// If we ever want to make more states, we can. But right now there will only be one state
typedef enum MenuState {
        MENU_STATE_MAIN,
        MENU_STATE_IMG_SELECT,
        MENU_STATE_HEIGHTMAP_EDIT,
        NUM_STATES
} MenuState;


/// @brief 
typedef struct MenuOptions {
        struct nk_context *ctx; //context for nuklear window
        struct nk_glfw glfw;    //rendering data for the menu
        MenuState state;        //what window the program is in

        int font_size; //will govern the size of the letters and textboxes in the gui


        char img_path[GUI_IMG_PATH_BUFF_LEN]; //buffer where the image path is stored. (textbox buffer)
        float img_aspect_ratio; //the aspect ratio of the image (width / height)
        GLuint img_tex;   //glfw texture id of the image we want to use
        struct nk_image img_nk; //nk handler of the image
        int img_copied; //boolean for if the image is being used by cam_proj_mdl (borrowed)


        GLuint ecam_tex;        //texture to the framebuffer of what the editor camera sees
        struct nk_image ecam_tex_nk; //nk handler of the editor tex
        vec2s ecam_offset;     //how far away the editor camera is from the center of the image (x and y correspond with cam_right and cam_up)


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




/// @brief This is a drawcall for the gui menu
/// @param wnd window to render the menu onto
/// @param delta_time the delta time
/// @param gui_menu gui menu
void nuklear_menu_render(GLFWwindow *wnd, float delta_time, MenuOptions *const gui_menu);


#endif