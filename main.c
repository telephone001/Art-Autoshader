#include <stdio.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_include.h>

#include "src/gui.h"
#include "src/editor.h"
#include "src/light_sources.h"
#include "src/heighttracer_cpu.h"
#include "src/heighttracer_cuda.cuh"

#include "general/debug.h"
#include "opengl_util.h"
#include "opengl_texture_util.h"
#include "glfw_window.h"

#include <cglm/mat4.h>
#include <cglm/cam.h>
#include <cglm/util.h>
#include <cglm/struct.h>
#include <cglm/io.h>

#define SCR_LENGTH 800
#define SCR_HEIGHT 800
#define MAX_EDITORS 100
#define ORTHO_SCALE (1.0f / 100.0f)

extern Camera camera; // handler for camera data
extern int in_menu; // menu status

typedef enum DebugThing {
    DEBUG_NONE,
    SPAWN_EDITOR,
    DELETE_EVERYTHING,
    SPAWN_POINT_LIGHT,
    SPAWN_DIRECTIONAL_LIGHT,
    SPAWN_LIGHTS_FOR_CAMERA_RAYS,
    SPAWN_LIGHTS_FOR_INTERSECTIONS,
    CPU_TEST_RAYTRACE,
    CUDA_TEST_RAYTRACE
} DebugThing;

DebugThing debug_thing = DEBUG_NONE;

mat4 flycam_projection;
mat4 flycam_view;
mat4 offset_view;
mat4 ortho_proj;

// Convenience macros
#define V3_X(v) ((v).raw[0])
#define V3_Y(v) ((v).raw[1])
#define V3_Z(v) ((v).raw[2])
#define V2_X(v) ((v).raw[0])
#define V2_Y(v) ((v).raw[1])

void opengl_settings_init() {
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glm_perspective(glm_rad(FOVY), (float)SCR_LENGTH / SCR_HEIGHT, NEAR, FAR, flycam_projection);

    glm_ortho(
        -SCR_LENGTH * ORTHO_SCALE, SCR_LENGTH * ORTHO_SCALE,
        -SCR_HEIGHT * ORTHO_SCALE, SCR_HEIGHT * ORTHO_SCALE,
        NEAR, FAR, ortho_proj
    );

    glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
}

void key_callback_menu_switching(GLFWwindow* wnd, int key, int scancode, int action, int mods) {
    if (in_menu) {
        glfwSetScrollCallback(wnd, nk_gflw3_scroll_callback);
        glfwSetCharCallback(wnd, nk_glfw3_char_callback);
        glfwSetMouseButtonCallback(wnd, nk_glfw3_mouse_button_callback);
        nk_glfw3_key_callback(wnd, key, scancode, action, mods);
    }
    else {
        glfwSetCursorPosCallback(wnd, mouse_callback);
        glfwSetScrollCallback(wnd, NULL);
        glfwSetCharCallback(wnd, NULL);
        glfwSetMouseButtonCallback(wnd, NULL);

        if (key == GLFW_KEY_LEFT_SHIFT) {
            switch (action) {
            case GLFW_PRESS: camera.speed = NORMAL_CAMERA_SPEED / 5; break;
            case GLFW_RELEASE: camera.speed = NORMAL_CAMERA_SPEED; break;
            }
        }
        if (key == GLFW_KEY_LEFT_CONTROL) {
            switch (action) {
            case GLFW_PRESS: camera.speed = NORMAL_CAMERA_SPEED * 3; break;
            case GLFW_RELEASE: camera.speed = NORMAL_CAMERA_SPEED; break;
            }
        }

        if (key == GLFW_KEY_P && action == GLFW_PRESS) debug_thing = SPAWN_EDITOR;
        if (key == GLFW_KEY_L && action == GLFW_PRESS) debug_thing = DELETE_EVERYTHING;
        if (key == GLFW_KEY_1 && action == GLFW_PRESS) debug_thing = SPAWN_POINT_LIGHT;
        if (key == GLFW_KEY_2 && action == GLFW_PRESS) debug_thing = SPAWN_DIRECTIONAL_LIGHT;
        if (key == GLFW_KEY_5 && action == GLFW_PRESS) debug_thing = CPU_TEST_RAYTRACE;
        if (key == GLFW_KEY_6 && action == GLFW_PRESS) debug_thing = CUDA_TEST_RAYTRACE;
    }

    // ESC key menu toggle
    static double last_x_pos, last_y_pos;
    if (key == GLFW_KEY_ESCAPE && action != GLFW_PRESS) {
        if (in_menu) {
            glfwSetInputMode(wnd, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            glfwSetCursorPos(wnd, last_x_pos, last_y_pos);
        }
        else {
            glfwGetCursorPos(wnd, &last_x_pos, &last_y_pos);
            glfwSetInputMode(wnd, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        in_menu = !in_menu;
    }
}

void framebuffer_size_callback(GLFWwindow* const window, int width, int height) {
    glm_perspective(glm_rad(FOVY), (double)width / (double)height, NEAR, FAR, flycam_projection);
    glm_ortho(
        -width * ORTHO_SCALE, width * ORTHO_SCALE,
        -height * ORTHO_SCALE, height * ORTHO_SCALE,
        NEAR, FAR, ortho_proj
    );
    glViewport(0, 0, width, height);
}

int main() {
    if (!glfwInit()) return -1;

    GLFWwindow* wnd = glfw_setup(3, 3, SCR_LENGTH, SCR_HEIGHT, "art autoshader");
    if (!wnd) return -1;

    opengl_settings_init();

    GLuint shader_cam_proj = create_shader_program("shaders/cam_proj.vert", "shaders/cam_proj.frag", NULL, NULL, NULL);
    GLuint shader_cam_plane = create_shader_program("shaders/cam_plane.vert", "shaders/cam_plane.frag", NULL, NULL, NULL);
    GLuint shader_hmap = create_shader_program("shaders/hmap.vert", "shaders/hmap.frag", NULL, NULL, NULL);
    GLuint shader_light_sources = create_shader_program("shaders/light_source.vert", "shaders/light_source.frag", NULL, NULL, NULL);

    MenuOptions gui_menu;
    int err = nuklear_menu_init(&gui_menu, wnd, "fonts/american-typewriter.ttf", 22);
    ERR_ASSERT_RET((err >= 0), -1, "nuklear window could not be created");

    glfwSetKeyCallback(wnd, key_callback_menu_switching);

    LightSourcesData light_sources_data = { 0 };
    GLuint point_light_tex, directional_light_tex;
    load_2dtexture(&point_light_tex, "textures/point_light.jpg", GL_RGB);
    load_2dtexture(&directional_light_tex, "textures/directional_light.jpg", GL_RGB);

    err = light_sources_rd_init(&(light_sources_data.rd), shader_light_sources, point_light_tex, directional_light_tex);
    ERR_ASSERT_RET((err >= 0), -2, "light sources struct could not be created");

    Editor editors[MAX_EDITORS] = { 0 };
    int cnt = 0;

    glfwSetInputMode(wnd, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    GL_PRINT_ERR();

    while (!glfwWindowShouldClose(wnd)) {
        static float last_frame;
        float t = glfwGetTime();
        float delta_time = t - last_frame;
        last_frame = t;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        get_cam_view(camera, flycam_view);

        // --- SPAWN EDITOR ---
        if (debug_thing == SPAWN_EDITOR) {
            if (editors[cnt].mdl_cam_plane.textures != NULL && gui_menu.img_tex == editors[cnt].mdl_cam_plane.textures[0])
                editors[cnt].mdl_cam_plane.textures[0] = 0;
            editor_free(&(editors[cnt]));
            int err = editor_init(&(editors[cnt]), wnd, &gui_menu, shader_cam_proj, shader_cam_plane, shader_hmap, 100, 100);
            hmap_edit_sinc(&(editors[cnt]));
            if (err < 0) {
                if (editors[cnt].mdl_cam_plane.textures != NULL && gui_menu.img_tex == editors[cnt].mdl_cam_plane.textures[0])
                    editors[cnt].mdl_cam_plane.textures[0] = 0;
                editor_free(&(editors[cnt]));
            }
            debug_thing = DEBUG_NONE;
            cnt++; if (cnt >= MAX_EDITORS) cnt = 0;
        }

        // --- DELETE EVERYTHING ---
        if (debug_thing == DELETE_EVERYTHING) {
            for (int i = 0; i < MAX_EDITORS; i++) {
                if (editors[i].mdl_cam_plane.textures != NULL && gui_menu.img_tex == editors[i].mdl_cam_plane.textures[0])
                    editors[i].mdl_cam_plane.textures[0] = 0;
                editor_free(&(editors[i]));
            }
            for (int i = 0; i < MAX_LIGHT_SOURCES; i++)
                light_sources_data.lights[i] = (LightSource){ 0 };
            debug_thing = DEBUG_NONE;
        }

        // --- SPAWN LIGHTS ---
        if (debug_thing == SPAWN_POINT_LIGHT) { light_source_add(&light_sources_data, (LightSource) { POINT, camera.pos, 0.5f }); debug_thing = DEBUG_NONE; }
        if (debug_thing == SPAWN_DIRECTIONAL_LIGHT) { light_source_add(&light_sources_data, (LightSource) { DIRECTIONAL, camera.pos, 1.0f }); debug_thing = DEBUG_NONE; }

        // --- CPU RAYTRACING ---
        if (debug_thing == CPU_TEST_RAYTRACE && editors[0].mdl_cam_proj.vao != 0) {
            int width, height;
            glfwGetWindowSize(wnd, &width, &height);
            vec3s* cam_dirs = ht_generate_camera_directions(&camera, width, height);

            double start_time = glfwGetTime(); // Start CPU timer
            int lights_added_cpu = 0;

            for (int i = 0; i < height; i += 50) {
                for (int j = 0; j < width; j += 50) {
                    float t_ray;
                    vec3s point;
                    int hit = ht_intersect_heightmap_ray(editors[0].hmap, editors[0].hmap_w, editors[0].hmap_l,
                        camera.pos, cam_dirs[i * width + j], 0.1f, 500.0f, &t_ray, &point);
                    if (hit) {
                        vec3s world_point = glms_vec3_add(camera.pos, glms_vec3_scale(cam_dirs[i * width + j], t_ray));
                        light_source_add(&light_sources_data, (LightSource) { POINT, world_point, 1.0f });
                        printf("CPU Hit at t=%.6f: %.6f %.6f %.6f\n", t_ray, V3_X(world_point), V3_Y(world_point), V3_Z(world_point));
                        lights_added_cpu++; // Increment CPU lights count
                    }
                }
            }
            double end_time = glfwGetTime(); // Stop CPU timer
            printf("CPU Raytracing finished: %d lights added. Time taken: %.6f seconds\n", lights_added_cpu, end_time - start_time);


            free(cam_dirs);
            debug_thing = DEBUG_NONE;
        }

        // --- CUDA RAYTRACING ---
        if (debug_thing == CUDA_TEST_RAYTRACE && editors[0].mdl_cam_proj.vao != 0) {
            int width, height;
            glfwGetWindowSize(wnd, &width, &height);

            float* t_results_cuda = NULL;
            vec3s* points_results_cuda = NULL;

            // Start CUDA timer (measures host-side call, including transfer and kernel launch)
            double start_time = glfwGetTime();

            ht_trace_all_cuda(
                editors[0].hmap, editors[0].hmap_w, editors[0].hmap_l,
                &camera, width, height, 0.1f, 500.0f,
                &t_results_cuda, &points_results_cuda
            );

            double end_time = glfwGetTime(); // Stop CUDA timer

            if (t_results_cuda && points_results_cuda) {
                int lights_added_cuda = 0; // Counter for lights spawned on host
                // Step safely over pixels (same step size as CPU version for comparison)
                int step = 50;
                if (width < step) step = 1;
                if (height < step) step = 1;

                for (int i = 0; i < height; i += step) {
                    for (int j = 0; j < width; j += step) {
                        int idx = i * width + j;

                        // Check for a valid hit (t >= 0.0f is considered a hit from the caller's perspective)
                        if (t_results_cuda[idx] >= 0.0f) {
                            vec3s hit_point = points_results_cuda[idx];
                            float hit_time = t_results_cuda[idx];

                            // Print every sampled hit point (like the CPU version)
                            fprintf(stdout, "CUDA Hit at t=%.6f: %.6f %.6f %.6f\n",
                                hit_time, V3_X(hit_point), V3_Y(hit_point), V3_Z(hit_point));

                            // Spawn point light if there is space
                            if (light_sources_data.num_lights < MAX_LIGHT_SOURCES) {
                                light_source_add(&light_sources_data, (LightSource) { POINT, hit_point, 1.0f });
                                lights_added_cuda++; // Increment CUDA lights count
                            }
                        }
                    }
                }

                // Summary print for CUDA
                printf("CUDA Raytracing finished: %d lights added. Time taken: %.6f seconds\n", lights_added_cuda, end_time - start_time);


                free(t_results_cuda);
                free(points_results_cuda);
            }
            else {
                fprintf(stderr, "CUDA Raytracing failed: no results returned\n");
            }

            debug_thing = DEBUG_NONE;
        }


        // --- RENDER EDITORS AND LIGHTS ---
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        for (int i = 0; i < MAX_EDITORS; i++) {
            if (editors[i].mdl_cam_proj.vao != 0)
                editor_render(&(editors[i]), 0, flycam_projection, flycam_view);
        }

        // --- FRAMEBUFFER TESTS ---
        GLint window_w, window_h;
        glfwGetWindowSize(wnd, &window_w, &window_h);
        glBindFramebuffer(GL_FRAMEBUFFER, gui_menu.ecam_data.fbo);
        glViewport(0, 0, gui_menu.ecam_data.width, gui_menu.ecam_data.height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        Camera editor_cam = camera;
        float ofx = V2_X(gui_menu.ecam_data.pos_offset);
        float ofy = V2_Y(gui_menu.ecam_data.pos_offset);
        editor_cam.pos = glms_vec3_add(editor_cam.pos, glms_vec3_scale(editor_cam.right, ofx));
        editor_cam.pos = glms_vec3_add(editor_cam.pos, glms_vec3_scale(editor_cam.up, ofy));
        get_cam_view(editor_cam, offset_view);

        for (int i = 0; i < MAX_EDITORS; i++) {
            if (editors[i].mdl_cam_proj.vao != 0)
                editor_render(&(editors[i]), 1, (gui_menu.ecam_data.in_perspective) ? flycam_projection : ortho_proj, offset_view);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, window_w, window_h);

        light_sources_render(&light_sources_data, flycam_projection, flycam_view);

        if (in_menu) {
            nuklear_menu_render(wnd, delta_time, &gui_menu);
            glfwSetCursorPosCallback(wnd, NULL);
        }
        else {
            handle_wasd_move(wnd, delta_time);
            glfwSetCursorPosCallback(wnd, mouse_callback);
        }

        glfwSwapBuffers(wnd);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}