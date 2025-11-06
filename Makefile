CC = gcc

myutil := ./new_myutility
lib := ./lib

# Nuklear/demo/glfw_opengl3 has been moved to newmyutility so nuklear doesn't change how our gui looks later
include := cglm/include glfw/include Nuklear stb glad/include


#for including all the necessary paths.
INCLUDE_ALL_DEP = $(addprefix -I./lib/,$(include)) -I./new_myutility

#for static library
libraries = -lglfw3 -lcglm 


#object files
util_files = opengl_util glfw_window general/debug opengl_texture_util
util_paths = $(addprefix $(myutil)/,$(util_files))
util_paths_obj = $(addsuffix .o,$(util_paths))

local_files = gui editor
local_paths = $(addprefix src/,$(local_files))
local_paths_obj = $(addsuffix .o,$(local_paths))

CFLAGS = -Wall $(util_paths_obj) $(local_paths_obj) $(lib)/glad.o $(INCLUDE_ALL_DEP) -I$(myutil) -L$(lib) $(libraries) -Wno-missing-braces



all: main.c $(util_paths_obj) $(local_paths_obj) $(lib)/glad.o
	$(CC) main.c $(CFLAGS) -o output && ./output.exe

#for compiling the glad library
$(lib)/glad.o: $(lib)/glad/src/glad.c
	$(CC) -c $(lib)/glad/src/glad.c -I$(lib)/glad/include -o $(lib)/glad.o
	
#generic for any other .o file
%.o: %.c %.h
	$(CC) -c -MMD -MP $< -o $@ $(INCLUDE_ALL_DEP) -L$(lib) -lcglm -I./new_myutility


clean_locals:
	rm -f $(local_paths_obj) $(local_paths_obj:.o=.d)
clean_util:
	rm -f $(util_paths_obj) $(util_paths_obj:.o=.d)
clean_lib:
	rm -f $(lib)/glad.o


clean: clean_util clean_lib clean_locals
	echo successfully cleaned!


-include $(util_paths_obj:.o=.d) $(local_paths_obj:.o=.d)

