cmake_minimum_required( VERSION 3.5 )

project( cmake_test )

set(CMAKE_C_STANDARD 11)




# where to put the .a files
set(LIB_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/libs)

add_library(mylib STATIC foo.c)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/libs)