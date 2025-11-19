#version 330 core

layout(location = 0) in float height;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform int hmap_row_len;

void main()
{
    int idx = gl_VertexID;
    int x = idx % hmap_row_len;
    int z = idx / hmap_row_len;

    vec3 pos = vec3(x, height, z);
    gl_Position = projection * view * model * vec4(pos, 1.0);
}
