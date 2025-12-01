#version 330

layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec2 a_tex;

uniform mat4 projection;
uniform mat4 view;
 

out vec2 tex;


void main()
{
        gl_Position = projection * view * vec4(a_pos, 1.0);
        tex = a_tex;
}
