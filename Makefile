CC = gcc

myutil := ./new_myutility
lib := ./lib

# Nuklear/demo/glfw_opengl3 has been moved to newmyutility so nuklear doesn't change how our gui looks later
include := cglm/include glfw/include Nuklear stb glad/include


#for including all the necessary paths.
INCLUDE_ALL_DEP = $(addprefix -I./lib/,$(include)) -I./new_myutility

#platform specific make for static libraries
ifeq ($(OS),Windows_NT)
	libraries = -lglfw3 -lcglm
else
	libraries = -lglfw3 -lcglm -ldl -lX11 -lpthread -lm 
endif

#object files
util_files = opengl_util glfw_window general/debug opengl_texture_util opengl_rbo_util
util_paths = $(addprefix $(myutil)/,$(util_files))
util_paths_obj = $(addsuffix .o,$(util_paths))

local_files = gui editor light_sources transform
local_paths = $(addprefix src/,$(local_files))
local_paths_obj = $(addsuffix .o,$(local_paths))

local_cpp_files = heighttracer_cpu
local_cpp_paths = $(addprefix src/,$(local_cpp_files))
local_cpp_paths_obj = $(addsuffix .o,$(local_cpp_paths))

LDFLAGS = $(util_paths_obj) $(local_paths_obj) $(lib)/glad.o $(local_cpp_paths_obj)
CFLAGS = -Wall $(INCLUDE_ALL_DEP) -I$(myutil) -L$(lib) $(libraries) -Wno-missing-braces


# link in C++
all: main.o $(local_cpp_paths_obj)
	g++ main.o $(LDFLAGS) $(CFLAGS) -o output && ./output.exe

main.o: main.c $(util_paths_obj) $(local_paths_obj) $(lib)/glad.o
	gcc -c main.c $(CFLAGS) -o main.o



#for compiling the glad library
$(lib)/glad.o: $(lib)/glad/src/glad.c
	$(CC) -c $(lib)/glad/src/glad.c -I$(lib)/glad/include -o $(lib)/glad.o
	

#generic for our c++ files
%.o: %.cpp %.h
	g++ -c -MMD -MP $< -o $@ $(INCLUDE_ALL_DEP)

#generic for any other .o file
%.o: %.c %.h
	$(CC) -c -MMD -MP $< -o $@ $(INCLUDE_ALL_DEP)


clean_locals:
	rm -f $(local_paths_obj) $(local_paths_obj:.o=.d)
	rm -f $(local_cpp_paths_obj) $(local_cpp_paths_obj:.o=.d)
clean_util:
	rm -f $(util_paths_obj) $(util_paths_obj:.o=.d)
clean_lib:
	rm -f $(lib)/glad.o
clean_mainio:
	rm -f main.o


clean: clean_util clean_lib clean_locals clean_mainio
	echo successfully cleaned!


-include $(util_paths_obj:.o=.d) $(local_paths_obj:.o=.d)

