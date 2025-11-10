#version 330 core

out vec4 frag_color;
  
in vec2 tex;

uniform sampler2D plane_texture;


void main()
{
        frag_color = texture(plane_texture, tex);
}