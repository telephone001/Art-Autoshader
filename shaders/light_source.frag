

#version 330 core

out vec4 frag_color;

uniform sampler2D tex_point;
uniform sampler2D tex_directional;

uniform int light_source_type;

#define POINT 1
#define DIRECTIONAL 2

in vec2 tex_coord;

void main()
{
        if (light_source_type == POINT) {
                frag_color = texture(tex_point, tex_coord);
        } else if (light_source_type == DIRECTIONAL) {
                frag_color = texture(tex_directional, tex_coord);
        }
        
        if (length(frag_color) >= sqrt(4) - 0.15) {
                discard;
        }
}