// Transform.h
#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct HeightmapTransform {
    glm::vec3 translation;    // move onto image plane
    glm::vec3 rotation;       // euler angles (radians)
    glm::vec3 scale;          // scale to match image resolution

    HeightmapTransform()
        : translation(0.0f), rotation(0.0f), scale(1.0f) {}

    glm::mat4 getMatrix() const {
        glm::mat4 M(1.0f);

        // apply translation
        M = glm::translate(M, translation);

        // apply rotation
        M = glm::rotate(M, rotation.x, glm::vec3(1,0,0));
        M = glm::rotate(M, rotation.y, glm::vec3(0,1,0));
        M = glm::rotate(M, rotation.z, glm::vec3(0,0,1));

        // apply scale
        M = glm::scale(M, scale);

        return M;
    }
};
