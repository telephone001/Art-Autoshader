#version 330

layout (location = 0) in vec3 a_pos;

layout (std140) uniform cameraData {
        mat4 projection;
        mat4 view;
};    


void main()
{
        gl_Position = projection * view * vec4(a_pos, 1.0);
}
