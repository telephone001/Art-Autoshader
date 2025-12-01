#version 330

layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec2 a_tex;

uniform mat4 projection;
uniform mat4 view;
 

out vec2 tex;

uniform int in_ecam_view;

void main()
{
        gl_Position = projection * view * vec4(a_pos, 1.0);
        tex = a_tex;

        //flip the camera so it renders upright in nuklear gui
        if (in_ecam_view != 0) {
                gl_Position.y *= -1;
        }
}
