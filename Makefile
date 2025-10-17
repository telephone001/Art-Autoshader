CC = gcc

myutil := ./myutility
include := ./include
lib := ./lib

libraries = -lglfw3 -lcglm

#object files
util_files = opengl_util glfw_window general/debug
util_paths = $(addprefix $(myutil)/,$(util_files))


util_paths_obj = $(addsuffix .o,$(util_paths))

CFLAGS = -Wall $(util_paths_obj) $(lib)/glad.o -I$(include) -I$(myutil) -L$(lib) $(libraries) -Wno-missing-braces

#platform specific make
ifeq ($(OS),Windows_NT)
all: main.c $(util_paths_obj) $(lib)/glad.o
	$(CC) main.c $(CFLAGS) -o output && output.exe

else
libraries = -lglfw3WSL -ldl -lX11 -lpthread -lm
all: main.c $(util_paths_obj) $(lib)/glad.o
	$(CC) main.c $(CFLAGS) -o output && ./output
endif


#for compiling the glad library
$(lib)/glad.o: $(lib)/glad.c $(include)/GLAD/glad.h
	$(CC) -c $(lib)/glad.c -I$(include) -o $@

#generic for any other .o file
%.o: %.c %.h
	$(CC) -c -MMD $< -o $@ -I$(include) -I$(myutil)



clean_util:
	rm -f $(util_paths_obj) $(util_paths_dep:.o=.d)
clean_lib:
	rm -f $(lib)/glad.o


clean: clean_util clean_lib
	echo successfully cleaned!


-include $(util_paths_obj:.o=.d)

