#version 450

layout(set = 0, binding = 0, std140) uniform SceneData {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 ambient;
    vec4 dirDirection;
    vec4 dirColor;
    ivec4 counts;
} scene;

layout(set = 1, binding = 0, std140) uniform ModelData {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;
    vec4 specular;
} modelData;

layout(location = 0) in vec3 inPosition;

void main() {
    gl_Position = scene.proj * scene.view * modelData.model * vec4(inPosition, 1.0);
}
