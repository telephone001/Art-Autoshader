#version 410 core

out vec4 frag_color;
in float height;


void main()
{
        float h = (height+1) / 2.0f;
        frag_color = vec4(h,h,1.0, 1.0);
}