#version 330 core

out vec4 frag_color;
in float f_height;

uniform float opacity;

//takes in a float and outputs a color on a spectrum.
//taken from a previous project
vec3 palette(float t) 
{
    	vec3 a = vec3(0.500, 0.500, 0.500);
        vec3 b = vec3(0.500, 0.500, 0.500);
        vec3 c = vec3(1.000, 1.000, 1.000);
        vec3 d = vec3(0.000, 0.333, 0.667);
    	return a + b*cos(6.28318*(c*t+d));
}

void main()
{
        float h = (f_height+1) / 5.0f;
        frag_color = vec4(palette(h * 0.2 - 0.05), opacity);
}