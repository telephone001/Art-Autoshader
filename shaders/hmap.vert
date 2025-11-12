#version 410 core
layout (location = 0) in float a_height;

layout (std140) uniform cameraData {
        mat4 projection;
        mat4 view;
};  

out float height;

uniform int hmap_row_len;

void main()
{       
        int x = gl_VertexID / hmap_row_len;
        int z = gl_VertexID % hmap_row_len;

        height = a_height;

        gl_Position = projection * view * vec4(vec3(x , a_height, z), 1.0);
}