#version 330

layout(location = 0) in float height;

layout (std140) uniform cameraData {
        mat4 projection;
        mat4 view; // DO NOT USE THIS VIEW
};    

//transformation matrix
uniform mat4 model;
uniform int hmap_row_len;

uniform mat4 offset_view;
uniform mat4 ortho_proj;

uniform int editor_mode;

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

        gl_Position = offset_view * model * vec4(pos, 1.0);

        //editor mode causes the 
        if (editor_mode != 0) {
                gl_Position *= ortho_proj;    
        } else {
                gl_Position *= projection;  
        }

}




