#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in float aHeight;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;        // ← NEW

out float heightValue;

void main()
{
    vec3 displaced = vec3(aPos.x, aPos.y, aPos.z + aHeight);

    gl_Position = projection * view * model * vec4(displaced, 1.0);
    heightValue = aHeight;
}
