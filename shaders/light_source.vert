
#version 330 core

layout (location = 0) in vec2 vertex_pos;

uniform mat4 projection;
uniform mat4 view;

uniform vec3 light_source_pos;
uniform vec3 cam_right;
uniform vec3 cam_up;

out vec2 tex_coord;

void main()
{          
        //we need to do a flip to fix the texture coordinates    
        vec2 flipped_vertex_pos = vec2(vertex_pos.x, vertex_pos.y * -1);
        tex_coord = clamp(flipped_vertex_pos, 0, 1);

        // this determines the size of the billboard
        vec2 trans_pos = vertex_pos * vec2(1);

        vec4 billboarded_world_pos = vec4((trans_pos.x * cam_right + trans_pos.y * cam_up) + light_source_pos, 1.0);
        gl_Position = projection * view * billboarded_world_pos;
}