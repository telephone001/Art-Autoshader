#version 330 core

layout(location = 0) in float height;

//transformation matrix
uniform mat4 model;
uniform int hmap_row_len;

uniform mat4 view;
uniform mat4 projection;

uniform int in_ecam_view;

// this will be sent to the fragment shader to color the heightmap for now
out float f_height;

void main()
{
        // this calculates the x and y coordinates
        int idx = gl_VertexID;
        int x = idx % hmap_row_len;
        int z = idx / hmap_row_len;
    
        //send height to hmap.frag
        f_height = height;
    
        //the heightmap vertices before the transformation
        vec3 pos = vec3(x, height, z);
    
        
        gl_Position = projection * view * model * vec4(pos, 1.0);
        
        //flip the camera so it renders upright in nuklear gui
        if (in_ecam_view != 0) {
                gl_Position.y *= -1;
        }
}
