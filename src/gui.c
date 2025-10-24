#define  NK_IMPLEMENTATION
#define  NK_GLFW_GL3_IMPLEMENTATION

#include "gui.h"

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
int nuklear_menu_init(MenuOptions *gui_menu, GLFWwindow *wnd, const char *const font_path, int font_size) 
{
	//initialize menu and also fill in some stuff
	*gui_menu = (MenuOptions){
		.ctx = nk_glfw3_init(wnd, NK_GLFW3_INSTALL_CALLBACKS),  //actually create the gui
		
		.state = MENU_STATE_MAIN,   
		
		.font_size = font_size,

		// 0 for no image selected (will modify when choosing an image)
		.img_path = {0}, //empty path
        	.img_tex = 0,	
        	.img_nk = {0},
	};

	ERR_ASSERT_RET((gui_menu->ctx != 0), -1, "nk_glfw3_init didn't work");


        //set up font
        struct nk_font_atlas* atlas;
        nk_glfw3_font_stash_begin(&atlas);
	ERR_ASSERT_RET((atlas != NULL), -2, "nk_glfw3_init didn't work");

        //set up font. last parameter = config and 0 = default config
    	struct nk_font *font = nk_font_atlas_add_from_file(atlas, font_path, font_size, 0);
	ERR_ASSERT_RET((font != NULL), -3, "font couldn't be added");


   	nk_glfw3_font_stash_end();

	nk_style_set_font(gui_menu->ctx, &font->handle);




	

	return 0;
}


// :::::::::::::::TODO:::::::::::::::::::

static int state_main_render(const MenuOptions *const gui_menu)
{
	//TODO you are ignoring a lot of potential error messages here but this is for prototype


	nk_layout_row_static(gui_menu->ctx, 30, 200, 1);
	nk_edit_string_zero_terminated(gui_menu->ctx, NK_EDIT_FIELD, gui_menu->img_path, GUI_IMG_PATH_BUFF_LEN, nk_filter_default);


	if (nk_button_label(gui_menu.ctx, "use image")) {
		nk_layout_row_static(gui_menu.ctx, 30, GUI_DRAWMODE_BOX_LENGTH, 1);
	}


	//allocate some menu area for the image then render the image of the wave
	nk_layout_row_begin(gui_menu->ctx, NK_STATIC, 150, 1);
        nk_layout_row_push(gui_menu->ctx, 150);
	nk_image(gui_menu->ctx, (gui_menu->img_nk));



	return 0;
}


/// @brief This is a drawcall for the gui menu
/// @param wnd window to render the menu onto
/// @param gui_menu gui menu
void nuklear_menu_render(GLFWwindow *wnd, MenuOptions *const gui_menu)
{	
	nk_glfw3_new_frame();
	
	//if can't make menu, cleanup and return. This cannot be turned into a return assert
	if (!nk_begin(gui_menu->ctx, "menu", nk_rect(0, 0, 225, 300),
		     NK_WINDOW_BORDER | 
		     NK_WINDOW_TITLE | 
		     NK_WINDOW_MINIMIZABLE | 
		     NK_WINDOW_MOVABLE | 
		     NK_WINDOW_SCALABLE)) {
		goto exit;
	}


	//render the stuff inside the menu
	state_main_render(gui_menu);


	//necessary code for rendering
	exit:

	nk_glfw3_render(NK_ANTI_ALIASING_ON, MAX_VERTEX_BUFFER, MAX_ELEMENT_BUFFER);
	nk_end(gui_menu->ctx);
}
