
#include "editor.h"

extern Camera camera;

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
)
{

        vec3s plane_pos = glms_vec3_add(cam.pos, glms_vec3_scale(cam.front, plane_dist));


        //the dist between rectangle origin and rectangle top
        float dist_up = tan(glm_rad(fovy)/2) * glms_vec3_norm(glms_vec3_sub(plane_pos, cam.pos));

        //the dist between rectangle origin and the rightost point of the rectangle
        float dist_right = dist_up * aspect_ratio;

        //find the four rectangle points:
        //vec3s r0 = plane_pos - cam_up * dist_up - cam.right * dist_right
        //vec3s r1 = plane_pos - cam_up * dist_up + cam.right * dist_right
        //vec3s r2 = plane_pos + cam_up * dist_up - cam.right * dist_right
        //vec3s r3 = plane_pos + cam_up * dist_up + cam.right * dist_right

        vec3s rect[4];
        for (int i = 0; i < 4; i++) {
                rect[i] = plane_pos;
        }
        
        //23
        //01 
        rect[0] = glms_vec3_add(rect[0], glms_vec3_scale(cam.up, -dist_up));
        rect[1] = glms_vec3_add(rect[1], glms_vec3_scale(cam.up, -dist_up));
        rect[2] = glms_vec3_add(rect[2], glms_vec3_scale(cam.up,  dist_up));
        rect[3] = glms_vec3_add(rect[3], glms_vec3_scale(cam.up,  dist_up));

        rect[0] = glms_vec3_add(rect[0], glms_vec3_scale(cam.right, -dist_right));
        rect[1] = glms_vec3_add(rect[1], glms_vec3_scale(cam.right,  dist_right));
        rect[2] = glms_vec3_add(rect[2], glms_vec3_scale(cam.right, -dist_right));
        rect[3] = glms_vec3_add(rect[3], glms_vec3_scale(cam.right,  dist_right));



        //we should have 5 points for our vertices. (the 4 above and the cam pos) 
        //Now we start filling in the renderdata

        *rdata = (RenderData) {
	        
                // will be filled out below
                .vao = 0, 
	        .vbo = 0, 
	        .ebo = 0, 

	        .vertices = malloc(sizeof(vec3s) * 5), //we only need 5 points
	        .vertices_stride = 3,                  //has to be 3 cuz of vec3
	        .vertices_length = 3 * 5,

	        .indices = malloc(sizeof(unsigned int) * 2 * 10), //10 because there are 10 ways to connect each of the 5 points
	        .indices_stride = 2, // 2 because we are rendering lines
	        .indices_length = 2 * 10,

	        .textures = NULL, //no textures
	        .num_textures = 0,    // nope.

	        .primitive_type = GL_LINES,
	        .shader = shader,
        };

        ERR_ASSERT_RET((rdata->vertices != NULL), -2, "malloc failed");
        ERR_ASSERT_RET((rdata->indices != NULL), -2, "malloc failed");

        //copy all points into renderdata. the first vec3 will be the cam pos
        memcpy(rdata->vertices, cam.pos.raw, sizeof(vec3s));
        memcpy(rdata->vertices + 3, (float*)rect, sizeof(vec3s) * 4);

        //make a line for each possible connection in the index buffer
        {
                int cnt = 0; //
                for (unsigned int i = 0; i < 5; i++) {
                        for (unsigned int j = i+1; j < 5; j++) {
                                rdata->indices[2 * cnt] = i;
                                rdata->indices[2 * cnt + 1] = j;
                                cnt++;
                        }
                }
        }


        

        bind_vao_and_vbo(&(rdata->vao), &(rdata->vbo), rdata->vertices, sizeof(float) * rdata->vertices_length, GL_STATIC_DRAW);
        
        ERR_ASSERT_RET((rdata->vao != 0), -3, "vao failed");
        ERR_ASSERT_RET((rdata->vbo != 0), -4, "vbo failed");


        bind_ebo(&(rdata->ebo), rdata->indices, sizeof(unsigned int) * rdata->indices_length, GL_STATIC_DRAW);

        ERR_ASSERT_RET((rdata->vao != 0), -5, "ebo failed");

        
	//position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, rdata->vertices_stride * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
        

        return 0;
}

/// @brief fits a quad with an aspect ratio into the cam plane. Returns results through output_fitted
/// @param output_fitted output quad. (array of 4 vec3s)
/// @param aspect_ratio aspect ratio of the thing you want to fit
/// @param bound the bounds to fit into (vertices of the cam plane)
///     bottom left
///     bottom right
///     top left
///     top right
/// @return negative values mean error
int cam_plane_img_fit(vec3s output_fitted[4], float aspect_ratio, vec3s bound[4])
{
        /// 0 bottom left
        /// 1 bottom right
        /// 2 top left
        /// 3 top right
        float bound_w = glms_vec3_distance(bound[0], bound[1]);
        float bound_h = glms_vec3_distance(bound[0], bound[2]);

        float r_width, r_height;
        int err = img_rect_fit(&r_width, &r_height, aspect_ratio, bound_w, bound_h);

        ERR_ASSERT_RET((err >= 0), -3, "img_rect_fit inside cam_plane_img_fit didn't work\n");

        if (fabs(r_width - bound_w) <= 0.0001) {
                //width has been fitted
                //squish the height to fit img
                vec3s plane_up = glms_vec3_normalize(glms_vec3_sub(bound[2], bound[0]));
                float squish_amnt = (bound_h-r_height) / 2;

                vec3s squish = glms_vec3_scale(plane_up, squish_amnt);

                //push the bottoms up
                output_fitted[0] = glms_vec3_add(bound[0], squish);
                output_fitted[1] = glms_vec3_add(bound[1], squish);

                //push the tops down
                output_fitted[2] = glms_vec3_sub(bound[2], squish);
                output_fitted[3] = glms_vec3_sub(bound[3], squish);
                

                
        } else {
                ///height has been fitted
                //squish the width to fit the img
                vec3s plane_right = glms_vec3_normalize(glms_vec3_sub(bound[1], bound[0]));

                float squish_amnt = (bound_w-r_width) / 2;

                vec3s squish = glms_vec3_scale(plane_right, squish_amnt);

                //push the lefts right
                output_fitted[0] = glms_vec3_add(bound[0], squish);
                output_fitted[2] = glms_vec3_sub(bound[2], squish);

                //push the rights left
                output_fitted[1] = glms_vec3_add(bound[1], squish);
                output_fitted[3] = glms_vec3_sub(bound[3], squish);
        }
        return 0;
}

/// @brief initializes a cam plane model (plane that includes the image but not the heightmap)
///            This is used inside the editor model to display the image on the plane
/// @param cam_plane the struct we output to hold the cam plane
/// @param cam_proj the struct we get the cam plane from. This is a cam_proj_mdl
/// @param shader shader for the plane containing a texture
/// @param img_tex the texture of the image
/// @return 
static int cam_plane_mdl_init(RenderData *cam_plane, RenderData *cam_proj, GLuint shader, GLuint img_tex) 
{
        // THESE MUST BE MALLOCED SO THAT FREEING THIS DOESN'T KILL THE PROGRAM
        *cam_plane = (RenderData) {
	        
                // will be filled out below
                .vao = 0, 
	        .vbo = 0, 
	        .ebo = 0, 

	        .vertices = malloc((sizeof(vec3) + sizeof(vec2)) * 4), //we only need 4 points. All from cam_mdl
	        .vertices_stride = 5,                  //has to be 5 cuz of vec3 for pos and 2 for texcord
	        .vertices_length = 5 * 4,

	        .indices = malloc(sizeof(unsigned int) * 3 * 2), // 3 for each of the 2 triangles
	        .indices_stride = 3, // 3 for triangle primitives
	        .indices_length = 3 * 2,

	        .textures = malloc(sizeof(GLuint)), //you gotta alloc memory so that the menu image isn't the same as the projected image 
	        .num_textures = 1,   

	        .primitive_type = GL_TRIANGLES,
	        .shader = shader,
        };

        //23
        //01 
        //squish the plane to fit the aspect ratio
        printf("%f %f %f\n", cam_proj->vertices[3], cam_proj->vertices[4], cam_proj->vertices[5]);

        cam_plane_img_fit(
                (vec3s*)(cam_plane->vertices), 
                img_aspect_ratio(img_tex), 
                (vec3s*)(cam_proj->vertices + 3)  //skip over the first vec3.
        );

        printf("%f %f %f\n", cam_plane->vertices[0], cam_plane->vertices[1], cam_plane->vertices[2]);


        //initialize texture coords for vertices
        {
                float tex_coords[] = {
                        0,1,
                        1,1,
                        0,0,
                        1,0,
                }; 
                for (int i = 0; i < 4; i++) {
                        memcpy(
                                cam_plane->vertices + i * cam_plane->vertices_stride + 3,
                                tex_coords + 2*i,
                                sizeof(vec2s)
                        );
                }
        }

        //copy indices to renderdata
        {
                unsigned int indices[] = {
                        0,1,2,
                        1,2,3,
                };

                memcpy(
                        cam_plane->indices,
                        indices,
                        sizeof(indices)
                );
        }
        *cam_plane->textures = img_tex;


        //now do regular opengl initializaiton
        bind_vao_and_vbo(&(cam_plane->vao), &(cam_plane->vbo), cam_plane->vertices, sizeof(float) * cam_plane->vertices_length, GL_STATIC_DRAW);
        
        ERR_ASSERT_RET((cam_plane->vao != 0), -3, "vao failed");
        ERR_ASSERT_RET((cam_plane->vbo != 0), -4, "vbo failed");


        bind_ebo(&(cam_plane->ebo), cam_plane->indices, sizeof(unsigned int) * cam_plane->indices_length, GL_STATIC_DRAW);

        ERR_ASSERT_RET((cam_plane->vao != 0), -5, "ebo failed");

        
	//position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, cam_plane->vertices_stride * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

        // texture attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, cam_plane->vertices_stride * sizeof(float), (void*)(sizeof(vec3s)));
	glEnableVertexAttribArray(1);

        return 0;
}


/// @brief Will allocate and calculate the indices matrix and the size of it for a heightmap of points
///             Modified from an old project (thanks John)
/// @param r_heightmap_indices the address of the pointer to the indices array (returned thing)
/// @param r_indices_length returned indices length 
/// @param grid_length the length of the grid (how many points are on each row)
/// @param grid_width the width of the grid (how many points are on each column) 
/// @return 0 for success. negative value for error
static int heightmap_indices_create(
        unsigned int **r_heightmap_indices,
        unsigned int *r_indices_length, 
        const int grid_length,
        const int grid_width
)
{
        ERR_ASSERT_RET((grid_length > 1), -1, "grid length was less than 2");
        ERR_ASSERT_RET((grid_width > 1), -2, "grid width was less than 2");

        // each quad of 4 points needs 2 triangles. each triangle needs 3 points.
	const size_t indices_length = (grid_length - 1) * (grid_width - 1) * 2 * 3;

	unsigned int *indices = malloc(indices_length * sizeof(unsigned int));
        ERR_ASSERT_RET((indices != NULL), -3, "could not allocate indices of heightmap. Out of memory");

	//index of indices array to put values in
	int cur_indices_idx = 0;
	//current point the edge of the square is on
	int cur_point_idx = 0;

	//loop gets indexes of 1 square at a time on the grid
	for (int z = 0; z < grid_length - 1; z++) {
		for (int x = 0; x < grid_width - 1; x++) {
			//first triangle
			indices[cur_indices_idx]   = cur_point_idx;
			indices[cur_indices_idx+1] = cur_point_idx + 1;
			indices[cur_indices_idx+2] = cur_point_idx + grid_width;

			//second triangle
			indices[cur_indices_idx+3] = cur_point_idx + 1;
			indices[cur_indices_idx+4] = cur_point_idx + grid_width;
			indices[cur_indices_idx+5] = cur_point_idx + grid_width + 1;
			
                        cur_indices_idx += 6;
			cur_point_idx++;
		}
		cur_point_idx++;
	}

	*r_indices_length = indices_length;
	*r_heightmap_indices = indices;

        return 0;
}


/// @brief initializes the heightmap, mallocing a vertices and indices array. The vertices will be all 0s
///             and the indices will be filled out
/// @param hmap_rd the returned heightmap
/// @param shader  shader for the heightmap (tesselator)
/// @param hmap_w heightmap length (how many array members are in each row)
/// @param hmap_l heightmap width (how many array members are in each column)
/// @return 0 for success. negative values for error
int heightmap_mdl_init(
        RenderData *hmap_rd,
        GLuint shader,
        int hmap_w,
        int hmap_l
)
{
        int err = 0;

        *hmap_rd = (RenderData) {
                // will be filled out below
                .vao = 0, 
	        .vbo = 0, 
	        .ebo = 0, 

	        .vertices = malloc(hmap_l * hmap_w * sizeof(float)), 
	        .vertices_stride = 1,                 
	        .vertices_length = hmap_l * hmap_w,

                //These will be filled out with the heightmap_indices_create
	        .indices = NULL, 
	        .indices_stride = 3,
	        .indices_length = -1,

                //No textures!
	        .textures = NULL, 
	        .num_textures = 0,   

	        .primitive_type = GL_TRIANGLES,
	        .shader = shader,
        };

        memset(hmap_rd->vertices, 0, hmap_rd->vertices_length * sizeof(float));


        err = heightmap_indices_create(&(hmap_rd->indices), &(hmap_rd->indices_length), hmap_l, hmap_w);
        ERR_ASSERT_RET((err >= 0), -1, "could not create heightmap idices");


        //do opengl rendering stuff (fill out vao and vbo and ebo)
        bind_vao_and_vbo(
                &(hmap_rd->vao),
                &(hmap_rd->vbo), 
                hmap_rd->vertices, 
                sizeof(float) * hmap_rd->vertices_length, 
                GL_STATIC_DRAW
        );
        ERR_ASSERT_RET((hmap_rd->vao != 0), -3, "vao failed");
        ERR_ASSERT_RET((hmap_rd->vbo != 0), -4, "vbo failed");

        bind_ebo(
                &(hmap_rd->ebo), 
                hmap_rd->indices, 
                sizeof(unsigned int) * hmap_rd->indices_length, 
                GL_STATIC_DRAW
        );
        ERR_ASSERT_RET((hmap_rd->vao != 0), -5, "ebo failed");

        //position attribute
	glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, hmap_rd->vertices_stride * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
        

        return 0;

}


/// @brief initializes an editor given a pointer to it
/// @param editor the editor you want to initialize
/// @param wnd the window (used to find the aspect ratio)
/// @param gui_menu the gui_menu. Includes the texture and also we tell the menu that we took its texture
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
)
{
        int err;

        // No texture. No editor
        ERR_ASSERT_RET((gui_menu->img_tex != 0), -2, "cam_plane_mdl_init failed (cannot make plane without texture)");


        //width and height of the window in pixels
        int width, height;
        glfwGetFramebufferSize(wnd, &width, &height);

        
        err = cam_proj_mdl_init(
                &editor->mdl_cam_proj,
                shader_cam_proj,
                (float)width/(float)height,
                FOVY,
                camera,
                CAM_PROJ_MDL_DIST  
        );
        ERR_ASSERT_RET((err == 0), -1, "cam_proj_mdl_init failed");

        err = cam_plane_mdl_init(
                &editor->mdl_cam_plane,
                &editor->mdl_cam_proj,
                shader_cam_plane,
                gui_menu->img_tex
        );
        ERR_ASSERT_RET((err == 0), -3, "cam_plane_mdl_init failed");

        //tell the gui menu that you used their image to create an editor
        gui_menu->img_copied = 1;

        editor->hmap_l = hmap_idx_l;       //Length corresponds to the x coordinate
        editor->hmap_w = hmap_idx_w;       //Width  correspodns to the z coordinate

        err = heightmap_mdl_init(
                &(editor->hmap_rd),
                shader_hmap,
                editor->hmap_l,
                editor->hmap_w
        );
        ERR_ASSERT_RET((err == 0), -4, "heightmap_mdl_init failed");

        editor->cam = camera;     //copy the current camera into the editor


        //Commented it out for now
	//	transform_init(&editor->hmap_transform);
	//	editor->hmap_transform.translation[2] = 0.01f;
        return 0;
}


/// @brief renders a cam_plane model
/// @param cam_plane_rdata the renderdata of the plane you want to render.
void cam_plane_mdl_render(RenderData *cam_plane_rdata)
{
        glBindVertexArray(cam_plane_rdata->vao);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, cam_plane_rdata->textures[0]);

	glUseProgram(cam_plane_rdata->shader);
	glDrawElements(cam_plane_rdata->primitive_type, cam_plane_rdata->indices_length, GL_UNSIGNED_INT, 0);
}


/// @brief simple command to render a cam_proj model. 
/// @param cam_proj_rdata the renderdata of the thing you want to render (has to be initialized first)
void cam_proj_mdl_render(RenderData *cam_proj_rdata)
{
        glBindVertexArray(cam_proj_rdata->vao);
	glUseProgram(cam_proj_rdata->shader);
	glDrawElements(cam_proj_rdata->primitive_type, cam_proj_rdata->indices_length, GL_UNSIGNED_INT, 0);
}


/// @brief renders heightmap
/// @param hmap_rdata the heightmap you want to render
/// @param hmap_row_len the length of each row in the heightmap
/// @param mat4 model the matrix model for linear algebra manipulation in heightmap
void hmap_render(RenderData *hmap_rdata, int hmap_row_len, mat4 model)
{
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        glBindVertexArray(hmap_rdata->vao);
        glUseProgram(hmap_rdata->shader);

        glUniform1i(glGetUniformLocation(hmap_rdata->shader, "hmap_row_len"), hmap_row_len);

        GLint modelLoc = glGetUniformLocation(hmap_rdata->shader, "model");
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, (float*)model);

        glDrawElements(hmap_rdata->primitive_type, hmap_rdata->indices_length, GL_UNSIGNED_INT, 0);

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}



/// @brief renders the editor object
/// @param editor 
/// @return 
int editor_render(Editor *editor)
{
        cam_proj_mdl_render(&(editor->mdl_cam_proj));
        cam_plane_mdl_render(&(editor->mdl_cam_plane));

        // Heightmap model transform
        
        
        // TODO: I put this as identity so I can compile the code. You can delete this
        mat4 model = GLM_MAT4_IDENTITY_INIT;
        //transform_get_matrix(&editor->hmap_transform, model);

        hmap_render(&(editor->hmap_rd), editor->hmap_l, model);

        return 0;
}

/// @brief frees the data of an editor 
///             (WARNING: It also deletes gltextures. If something else is using your texture, set the texture in the renderdata to 0)
/// @param editor the editor you want to free
void editor_free(Editor *editor)
{
        if (editor != NULL) {
                //the functions below only frees the stuff that isn't null
                render_data_free(&(editor->mdl_cam_proj));
                render_data_free(&(editor->mdl_cam_plane));
                render_data_free(&(editor->hmap_rd));
        }
}

/// @brief change the editor's heightmap to a sinc function. This function buffers the subdata to the vbo
/// @param editor the editor to change its heightmap
void hmap_edit_sinc(Editor *editor)
{       

        int ofst1 = (((rand() % 100) - 50));
        int ofst2 = (((rand() % 100) - 50));
        float ofst3 = ((rand() % 10))+ 10;
        float ofst4 = ((rand() % 10))+ 10;

        //set the heightmap to a sinc function
        for (int i = 0; i < editor->hmap_w; i++) {
                for (int j = 0; j < editor->hmap_l; j++) {
                        int a = i - editor->hmap_w/2 + ofst1;
                        int b = j - editor->hmap_l/2 + ofst2;
                        float c = 0.15 * sqrt(a*a+b*b*ofst3/ofst4);
                        float m = 10;
                        
                        if (a==0 && b==0) {
                                editor->hmap_rd.vertices[i * editor->hmap_w + j] = m;
                        } else {
                                editor->hmap_rd.vertices[i * editor->hmap_w + j] = m * sin(c) / c;
                        }
                }
        }

        glBindBuffer(GL_ARRAY_BUFFER, editor->hmap_rd.vbo);

        glBufferSubData(
                GL_ARRAY_BUFFER,
                0,
                editor->hmap_rd.vertices_length * sizeof(float),
                editor->hmap_rd.vertices
        );
        
        glBindBuffer(GL_ARRAY_BUFFER, 0);
}

