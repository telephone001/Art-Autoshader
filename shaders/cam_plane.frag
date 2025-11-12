#version 410 core

out vec4 frag_color;
  
in vec2 tex;

uniform sampler2D plane_texture;

in float color_dbg;


void main()
{
        frag_color = texture(plane_texture, tex);

        frag_color = frag_color + vec4(color_dbg,0,0,1);
}