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
		//THESE ARE CREATED LATER IN THIS FUNCTION
		.ctx = NULL, 
		.glfw = {0},
		
		.state = MENU_STATE_MAIN,   
		
		.font_size = font_size,

		// 0 for no image selected currently (will modify when choosing an image)
		.img_path = {0}, //empty path Put to all zeros to ensure the last one is a null terminator
		.img_aspect_ratio = 0,
        	.img_tex = 0,	
        	.img_nk = {0},
		.img_copied = 0,
	};
	
	gui_menu->ctx = nk_glfw3_init(&gui_menu->glfw, wnd, NK_GLFW3_INSTALL_CALLBACKS);

	ERR_ASSERT_RET((gui_menu->ctx != 0), -1, "nk_glfw3_init didn't work");

	strncpy(gui_menu->img_path, GUI_IMG_PATH_START, GUI_IMG_PATH_START_LEN);

        //set up font
        struct nk_font_atlas* atlas;
        nk_glfw3_font_stash_begin(&gui_menu->glfw, &atlas);
	ERR_ASSERT_RET((atlas != NULL), -2, "nk_glfw3_init didn't work");

        //set up font. last parameter = config and 0 = default config
    	struct nk_font *font = nk_font_atlas_add_from_file(atlas, font_path, font_size, 0);
	ERR_ASSERT_RET((font != NULL), -3, "font couldn't be added");


   	nk_glfw3_font_stash_end(&gui_menu->glfw);

	nk_style_set_font(gui_menu->ctx, &font->handle);




	

	return 0;
}


// :::::::::::::::TODO:::::::::::::::::::

static int state_main_render(MenuOptions *const gui_menu)
{
	// you could move this into the gui_menu if you wanted to, but for testing a static var is easier
	static int img_err = 0; //the error value of load_2dtexture

	nk_layout_row_static(gui_menu->ctx, 30, 200, 1);
	nk_flags textbox_event =  nk_edit_string_zero_terminated(
		gui_menu->ctx,
		NK_EDIT_FIELD, 
		gui_menu->img_path + GUI_IMG_PATH_START_IDX, 
		GUI_IMG_FILENAME_MAX_LEN, //total size of the buffer you edit (includes null terminator) 
		nk_filter_default
	);

	//if the textbox has been selected, remove the red error message in the gui
	if (textbox_event & NK_EDIT_ACTIVE) {
		// TODO ERROR HERE
		img_err = 0;
	}

	// if use_img button pressed, set the menu image to the thing pointed to the textbox
	if (nk_button_label(gui_menu->ctx, "use image")) {

		//delete the texture first but only if someone else isnt using it
		if (gui_menu->img_tex != 0) {
			if (gui_menu->img_copied == 0) {
				glDeleteTextures(1, &gui_menu->img_tex);
			}
			
			gui_menu->img_tex = 0;
			gui_menu->img_nk = (struct nk_image){0};
			//you don't have to free anything in img_nk because it is not allocated
		}

		img_err = load_2dtexture(&gui_menu->img_tex, gui_menu->img_path, GL_RGB);

		//if we loaded a texture, set the other parameters in the menuOptions
		if (img_err >= 0 && gui_menu->img_tex != 0) {
			// Set the aspect ratio
			glBindTexture(GL_TEXTURE_2D, gui_menu->img_tex);
			int w, h;
			glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
			glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);
			gui_menu->img_aspect_ratio = (float)w/(float)h;	

			// set the img_nk
			gui_menu->img_nk = nk_image_id(gui_menu->img_tex);
		}
	}


	if (img_err < 0) {
		// display an error message if texture not found
		nk_layout_row_static(gui_menu->ctx, 20, 180, 1);
		nk_style_push_color(gui_menu->ctx, &gui_menu->ctx->style.text.color, nk_rgb(255, 0, 0));
		nk_label(gui_menu->ctx, "Texture not found", NK_TEXT_LEFT);
		nk_style_pop_color(gui_menu->ctx);
	} else if (gui_menu->img_tex != 0) {
		//otherwise, display the image and set the struct fields

		printf("%f\n", gui_menu->img_aspect_ratio);
		//these determine image dimensions
		nk_layout_row_begin(gui_menu->ctx, NK_STATIC, 150, 1);		 // controls height
        	nk_layout_row_push(gui_menu->ctx, 150 * gui_menu->img_aspect_ratio); // controls width

		nk_image(gui_menu->ctx, gui_menu->img_nk);
	}

	return 0;
}


/// @brief This is a drawcall for the gui menu
/// @param wnd window to render the menu onto
/// @param gui_menu gui menu
void nuklear_menu_render(GLFWwindow *wnd, MenuOptions *const gui_menu)
{	
	nk_glfw3_new_frame(&gui_menu->glfw);
	
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

	nk_glfw3_render(&gui_menu->glfw, NK_ANTI_ALIASING_ON, MAX_VERTEX_BUFFER, MAX_ELEMENT_BUFFER);
	nk_end(gui_menu->ctx);
}
