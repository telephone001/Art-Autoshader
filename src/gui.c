#define  NK_IMPLEMENTATION
#define  NK_GLFW_GL3_IMPLEMENTATION

#include "gui.h"



/// @brief creates an ecam_data given the width and height of framebuffer
/// @param ret_ecam_data the ecam_data you want to create
/// @param width pixel width of framebuffer you want to make
/// @param height pixel height of framebuffer you want to make
/// @return 0 on success. -1 on gl error
int editor_cam_data_init(EditorCamData *ecam_data, int width, int height)
{
	*ecam_data = (EditorCamData){
	// initialized here
        	.pos_offset = (vec2s){0},  
		.width = width,
		.height = height,
		.in_perspective = 0,

	// will be set below
		.fbo = 0,
		.rbo = 0,
		.tex = 0,
        	.tex_nk = {0},

	};

        glGenFramebuffers(1, &ecam_data->fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, ecam_data->fbo);

        ecam_data->tex = fbo_tex_init(ecam_data->fbo, width, height);
        ecam_data->rbo = fbo_rbo_init(ecam_data->fbo, width, height);

	// Error checking
	// we cant use the ret assert here because we gotta free the gui menu first.
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		glDeleteFramebuffers(1, &ecam_data->fbo);
		glDeleteRenderbuffers(1, &ecam_data->rbo);
		glDeleteTextures(1, &ecam_data->tex);
		
		return -1;
	} 

	ecam_data->tex_nk = nk_image_id(ecam_data->tex);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	return 0;
}

/// @brief frees the editor cam data
/// @param ecam_data  editor cam data
void editor_cam_data_free(EditorCamData *ecam_data)
{
	glDeleteFramebuffers(1, &ecam_data->fbo);
	ecam_data->fbo = 0;

	glDeleteTextures(1, &ecam_data->tex);
	ecam_data->tex = 0;

	glDeleteRenderbuffers(1, &ecam_data->rbo);
	ecam_data->rbo = 0;
}


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
int nuklear_menu_init(
	MenuOptions *gui_menu, 
	GLFWwindow *wnd, 
	const char *const font_path, 
	int font_size
) 
{
	//initialize menu and also fill in some stuff
	*gui_menu = (MenuOptions){
	// These are set right now
		.state = MENU_STATE_MAIN,   
		.font_size = font_size,

	// These are created in this function below
		.ctx = NULL, 
		.glfw = {0},

		.ecam_data = {0}, 

		// 0 for no image selected currently (will modify when choosing an image)
		.img_path = {0}, //empty path Put to all zeros to ensure the last one is a null terminator
		
	
	// worry about these later, as they are set in other functions
		.img_aspect_ratio = 0,
        	.img_tex = 0,	
        	.img_nk = {0},
		.img_copied = 0,

		
	};
	
	gui_menu->ctx = nk_glfw3_init(&gui_menu->glfw, wnd, NK_GLFW3_INSTALL_CALLBACKS);
	ERR_ASSERT_RET((gui_menu->ctx != 0), -1, "nk_glfw3_init didn't work\n");

	strncpy(gui_menu->img_path, GUI_IMG_PATH_START, GUI_IMG_PATH_START_LEN);

        //set up font
        struct nk_font_atlas* atlas;
        nk_glfw3_font_stash_begin(&gui_menu->glfw, &atlas);
	ERR_ASSERT_RET((atlas != NULL), -2, "nk_glfw3_init didn't work\n");

        //set up font. last parameter = config and 0 = default config
    	struct nk_font *font = nk_font_atlas_add_from_file(atlas, font_path, font_size, 0);
	ERR_ASSERT_RET((font != NULL), -3, "font couldn't be added\n");


   	nk_glfw3_font_stash_end(&gui_menu->glfw);

	nk_style_set_font(gui_menu->ctx, &font->handle);


	
	int width, height;
	glfwGetWindowSize(wnd, &width, &height);

	int err = editor_cam_data_init(&gui_menu->ecam_data, width, height);
	ERR_ASSERT_RET((err == 0), -4, "framebuffer gl error!\n");


	

	return 0;
}




/// @brief dynamically fits a nk image into the reminaing space of a gui menu.
///		you need to leave a vertical margin, which is for the other widgits
/// @param ctx the context of the nuklear menu
/// @param img the nuklear image
/// @param aspect_ratio the aspect ratio of the image
/// @param margin how much height space should I leave for the rest of the widgits in the gui
static void menu_fit_img(struct nk_context *ctx, struct nk_image img, float aspect_ratio, float margin)
{
	//total size of the gui menu
	struct nk_rect total = nk_window_get_content_region(ctx);

	float w,h;
	//TODO: 120 is used to fit the image into a bound. (used to calculate remaining space in gui)
	img_rect_fit(&w, &h, aspect_ratio, total.w, total.h - margin);

	//these determine image dimensions
	nk_layout_row_begin(ctx, NK_STATIC, h, 1); // controls height
        nk_layout_row_push(ctx, w);                // controls width

	nk_image(ctx, img);

}


//This thing is not used, but I thought it would be useful one day
/// @brief stretches an image to fit to the nuklear menu.
///		you need to leave a vertical margin, which is for the other widgits you put vertically
/// @param ctx the context of the nuklear menu
/// @param img the nuklear image
/// @param margin how much height space should I leave for the rest of the widgits in the gui
static void menu_stretch_img(struct nk_context *ctx, struct nk_image img, float margin)
{
	//total size of the gui menu
	struct nk_rect total = nk_window_get_content_region(ctx);

	float w = total.w;
	float h = total.h - margin;

	//these determine image dimensions
	nk_layout_row_begin(ctx, NK_STATIC, h, 1); // controls height
        nk_layout_row_push(ctx, w);                // controls width

	nk_image(ctx, img);
}

// :::::::::::::::TODO:::::::::::::::::::

/// @brief Renders the image selection state of the gui menu
/// @param gui_menu the gui menu's options
/// @return negative if there is an error. Positive for success
static int state_img_select_render(MenuOptions *const gui_menu)
{
	nk_layout_row_dynamic(gui_menu->ctx, 30, 1);
	if (nk_button_label(gui_menu->ctx, "back to main menu")) {
		gui_menu->state = MENU_STATE_MAIN;
	}

	// you could move this into the gui_menu if you wanted to, but for testing a static var is easier
	static int img_err = 0; //the error value of load_2dtexture

	nk_layout_row_dynamic(gui_menu->ctx, 30, 1);
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
			gui_menu->img_aspect_ratio = img_aspect_ratio(gui_menu->img_tex);	

			// set the img_nk
			gui_menu->img_nk = nk_image_id(gui_menu->img_tex);
		}
	}


	if (img_err < 0) {
		// display an error message if texture not found
		nk_layout_row_dynamic(gui_menu->ctx, 30, 1);
		nk_style_push_color(gui_menu->ctx, &gui_menu->ctx->style.text.color, nk_rgb(255, 0, 0));
		nk_label(gui_menu->ctx, "Texture not found", NK_TEXT_LEFT);
		nk_style_pop_color(gui_menu->ctx);
	} else if (gui_menu->img_tex != 0) {
		//otherwise, display the image
		menu_fit_img(gui_menu->ctx, gui_menu->img_nk, gui_menu->img_aspect_ratio, 120);
	}

	return 0;
}


static void state_main_render(MenuOptions *const gui_menu)
{
	nk_layout_row_dynamic(gui_menu->ctx, 30, 1);
	if (nk_button_label(gui_menu->ctx, "create image editor")) {
		gui_menu->state = MENU_STATE_IMG_SELECT;
	}

	nk_layout_row_dynamic(gui_menu->ctx, 30, 1);
	if (nk_button_label(gui_menu->ctx, "edit heightmap")) {
		gui_menu->state = MENU_STATE_HEIGHTMAP_EDIT;
	}
}


static void state_heightmap_edit_render(MenuOptions *const gui_menu, float delta_time, GLFWwindow *wnd)
{
	nk_layout_row_dynamic(gui_menu->ctx, 30, 1);
	if (nk_button_label(gui_menu->ctx, "back to main menu")) {
		gui_menu->state = MENU_STATE_MAIN;
	}

	nk_layout_row_dynamic(gui_menu->ctx, 30, 1);
	if (nk_button_label(gui_menu->ctx, "reset offset")) {
		gui_menu->ecam_data.pos_offset.x = 0;
		gui_menu->ecam_data.pos_offset.y = 0;
	}

	nk_layout_row_dynamic(gui_menu->ctx, 30, 1);
	nk_checkbox_label(gui_menu->ctx, "perspective", &gui_menu->ecam_data.in_perspective);
		
	int width, height;
	glfwGetWindowSize(wnd, &width, &height);

	// Replace the fbo if the nuklear window changed size
	if (width != gui_menu->ecam_data.width || height != gui_menu->ecam_data.height) {
		editor_cam_data_free(&gui_menu->ecam_data);
		//TODO remember the error value here
		editor_cam_data_init(&gui_menu->ecam_data, width, height);

		gui_menu->ecam_data.width = width;
		gui_menu->ecam_data.height = height;
	}


	if (gui_menu->ecam_data.tex != 0) {
		menu_fit_img(
			gui_menu->ctx, 
			gui_menu->ecam_data.tex_nk, 
			(float)gui_menu->ecam_data.width / (float)gui_menu->ecam_data.height, 
			100
		);
	} else {
		// display an error message if texture not found
		nk_layout_row_dynamic(gui_menu->ctx, 30, 1);
		nk_style_push_color(gui_menu->ctx, &gui_menu->ctx->style.text.color, nk_rgb(255, 0, 0));
		nk_label(gui_menu->ctx, "editor not selected.", NK_TEXT_LEFT);
		nk_style_pop_color(gui_menu->ctx);
	}

	// Get the rectangle of the img (prev widget)
	struct nk_rect img_rect = nk_widget_bounds(gui_menu->ctx);

	// NUKLEAR BUG. 
	img_rect.y -= img_rect.h;


	static double prev_mouse_x, prev_mouse_y;
	double mouse_x, mouse_y;

	// handle mouse behaviors in editor
	if (nk_input_is_mouse_hovering_rect(&gui_menu->ctx->input, img_rect)) {

		glfwGetCursorPos(wnd, &mouse_x, &mouse_y);

		int pressed = glfwGetMouseButton(wnd, GLFW_MOUSE_BUTTON_LEFT);

		//TODO: change the scale
		float scale = 5000;

		if ((pressed == GLFW_PRESS)) {
			gui_menu->ecam_data.pos_offset.x -= (mouse_x - prev_mouse_x) * delta_time * scale / img_rect.w;
			gui_menu->ecam_data.pos_offset.y += (mouse_y - prev_mouse_y) * delta_time * scale / img_rect.h;
		}

		//printf("%f %f\n", gui_menu->ecam_offset.x, gui_menu->ecam_offset.y);

		prev_mouse_x = mouse_x;
		prev_mouse_y = mouse_y;
	}

}


/// @brief This is a drawcall for the gui menu
/// @param wnd window to render the menu onto
/// @param delta_time the delta time
/// @param gui_menu gui menu
void nuklear_menu_render(GLFWwindow *wnd, float delta_time, MenuOptions *const gui_menu)
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




	//renders gui pages
	switch (gui_menu->state) {
		case MENU_STATE_MAIN:
			state_main_render(gui_menu);
			break;

		case MENU_STATE_IMG_SELECT:
			state_img_select_render(gui_menu);
			break;

		case MENU_STATE_HEIGHTMAP_EDIT:
			state_heightmap_edit_render(gui_menu, delta_time, wnd);
			break;

		default:
			nk_layout_row_static(gui_menu->ctx, 30, 200, 1);
			nk_label(gui_menu->ctx, "Error!", NK_TEXT_LEFT);
			break;
	}


	//necessary code for rendering
	exit:

	nk_glfw3_render(&gui_menu->glfw, NK_ANTI_ALIASING_ON, MAX_VERTEX_BUFFER, MAX_ELEMENT_BUFFER);
	nk_end(gui_menu->ctx);
}
