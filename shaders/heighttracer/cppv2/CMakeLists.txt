cmake_minimum_required(VERSION 3.10)
project(heighttracer_v2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------
# Path to engine source files
# ------------------------------------------------------
set(ENGINE_SRC_DIR ${CMAKE_SOURCE_DIR}/src)

# ------------------------------------------------------
# Include engine headers (editor.h, gui.h, etc.)
# ------------------------------------------------------
include_directories(${ENGINE_SRC_DIR})

# ------------------------------------------------------
# Add the executable
# ------------------------------------------------------
add_executable(heighttracer_v2
    mainv2.cpp
)

# ------------------------------------------------------
# If you need engine .c files linked in:
# (Editor depends on C code!)
# ------------------------------------------------------
add_library(engine_lib STATIC
    ${ENGINE_SRC_DIR}/editor.c
    ${ENGINE_SRC_DIR}/transform.c
    ${ENGINE_SRC_DIR}/opengl_util.c
    ${ENGINE_SRC_DIR}/glfw_window.c
    ${ENGINE_SRC_DIR}/opengl_texture_util.c
    ${ENGINE_SRC_DIR}/general/debug.c
)

# Ensure engine headers are visible
target_include_directories(engine_lib PUBLIC ${ENGINE_SRC_DIR})

# Link engine library into executable
target_link_libraries(heighttracer_v2 engine_lib)

# ------------------------------------------------------
# Windows / MinGW specific (avoid console warnings)
# ------------------------------------------------------
if (MINGW)
    target_link_libraries(heighttracer_v2
        -static
        -static-libgcc
        -static-libstdc++
    )
endif()
