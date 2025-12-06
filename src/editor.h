#ifndef __EDITOR_H__
#define __EDITOR_H__

#include "transform.h"

#include "gui.h"

#include "general/debug.h"

#include "glfw_window.h"
#include "opengl_util.h"



#define CAM_PROJ_MDL_DIST 10


typedef struct Editor {
        RenderData mdl_cam_plane;       //the model for the camera plane
        RenderData mdl_cam_proj;        //the model for the camera projection

        int hmap_w;       //Width corresponds to the x coordinate (size of each row)
        int hmap_l;       //Length  correspodns to the z coordinate
        float *hmap;
        RenderData hmap_rd;       //includes the heightmap as vertices

        Camera cam; //the camera object associated with the editor

        Transform hmap_transform; //matrix for transorming heightmap

} Editor;


/// @brief will output the renderdata of a wireframe of a camera projection. Note that this renderdata allocates 
///             memory for its vertex buffer! make sure to free it!!!
/// @param rdata the outputted render data for the wireframe of a camera projection
/// @param shader the shader is a parameter because I don't want to create a new shader object for each model
/// @param fovy the fovy of the camera.
/// @param aspect_ratio the aspect ratio of the camera width/height.
/// @param cam the camera object that you want to make a model of
/// @param plane_dist the distance between the camera and the plane looked at
/// @return 0 on success negatives on error
int cam_proj_mdl_init(
        RenderData *rdata, 
        GLuint shader,
        float aspect_ratio, 
        float fovy, 
        Camera cam,
        float plane_dist
);

/// @brief renders a cam_plane model
/// @param cam_plane_rdata the renderdata of the plane you want to render.
/// @param in_ecam_view
/// @param projection the projection matrix
/// @param view the view matrix
void cam_plane_mdl_render(RenderData *cam_plane_rdata, int in_ecam_view, mat4 projection, mat4 view);

/// @brief simple command to render a cam_proj model. 
/// @param cam_proj_rdata the renderdata of the thing you want to render (has to be initialized first)
/// @param projection the projection matrix
/// @param view the view matrix
void cam_proj_mdl_render(RenderData *cam_proj_rdata, mat4 projection, mat4 view);

/// @brief renders heightmap
/// @param hmap_rdata the heightmap you want to render
/// @param hmap_row_len the length of each row in the heightmap
/// @param in_ecam_view if we are rendering inside 
/// @param projection the projection matrix
/// @param view the view matrix
/// @param model the transformation matrix 
void hmap_render(RenderData *hmap_rdata, int hmap_row_len, int in_ecam_view, mat4 projection, mat4 view, mat4 model);

/// @brief uses an initialized hmap to build the heightmap rd, mallocing an indices array.
///             and the indices will be filled out
/// @param hmap_rd the returned heightmap
/// @param shader  shader for the heightmap (tesselator)
/// @param hmap_w heightmap length (how many array members are in each row)
/// @param hmap_l heightmap width (how many array members are in each column)
/// @param hmap the float array for the heightmap. (must be allocated before using this function)
/// @return 0 for success. negative values for error
int heightmap_mdl_init(
        RenderData *hmap_rd,
        GLuint shader,
        int hmap_w,
        int hmap_l,
        float *hmap
);

/// @brief initializes an editor given a pointer to it
/// @param editor the editor you want to initialize
/// @param wnd the window (used to find the aspect ratio)
/// @param gui_menu pointer to the gui_menu. Includes the texture and also we tell the menu that we took its texture
/// @param shader_cam_proj the shader for the cam projection wireframe
/// @param shader_cam_plane the shader for the plane containing the texture
/// @param shader_hmap the shader for rendering heightmap (tesselation)
/// @param hmap_idx_w for heightmap, how many array elements are in each row (x coordinate)
/// @param hmap_idx_l for heightmap, how many array elements are in each column (z coordinate)
/// @return 0 if there are no errors. Negative values for errors
int editor_init(
        Editor *editor, 
        GLFWwindow *wnd, 
        MenuOptions *gui_menu, 
        GLuint shader_cam_proj, 
        GLuint shader_cam_plane,
        GLuint shader_hmap,
        int hmap_idx_l,
        int hmap_idx_w
);

void editor_render(Editor *editor, int in_ecam_view, mat4 projection, mat4 view);

/// @brief deletes a camera plane renderdata.
/// @param cam_plane_rd camera plane renderdata
/// @param gl_delete_texture 0 if you don't want to remove the texture from GL state machine
void cam_plane_free(RenderData *cam_plane_rd, int gl_delete_texture);

/// @brief frees the data of an editor. Doesnt take into account the other editors who also may have the same texture
///             (WARNING: It also deletes gltextures. If something else is using your texture, set the texture in the renderdata to 0)
/// @param editor the editor you want to free
/// @param delete_texture 0 if you don't want to remove the texture from GL state machine
void editor_free_forced(Editor *editor, int gl_delete_texture);

/// @brief frees an editor "safely" in that if another editor uses the same texture, this function will not free
///             the texture of the editor we want to delete.
///             This is a little slower than editor_forced because we have to iterate through the editors
/// @param editors the array of all editors
/// @param idx_delete the index into the array of the editor you want to delete
/// @param editors_length the length of the array passed in.
///                             you can put a smaller amount to iterate through less editors.
void editor_free_safe(Editor *editors, int idx_delete, int editors_length);


/// @brief change the editor's heightmap to a sinc function. This function buffers the subdata to the vbo
/// @param editor the editor to change its heightmap
void hmap_edit_sinc(Editor *editor);

/// @brief This is the same heightmap thing from the parallel implementation
/// @param editor the editor you want to put this heightmap onto
void hmap_edit_test1(Editor *editor);

/// @brief puts the heightmap to all zeros
/// @param editor the editor you want to put this heightmap onto
void hmap_edit_zero(Editor *editor);

//
// Simple accessors for standalone CPU/CUDA heightmap usage
//

/// @brief Returns pointer to heightmap array
/// @note The returned pointer is owned by Editor. Do NOT free it.
float* editor_get_heightmap(Editor* editor);

/// @brief Returns heightmap width (number of X samples)
int editor_get_width(Editor* editor);

/// @brief Returns heightmap length (number of Z samples)
int editor_get_height(Editor* editor);

// modify heightmap (your preset)
void hmap_edit_edbert(Editor *editor);









#endif
