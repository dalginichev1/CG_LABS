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
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;

void main() {
    vec4 world = modelData.model * vec4(inPosition, 1.0);
    worldPosition = world.xyz;
    worldNormal = normalize(mat3(modelData.normalMatrix) * inNormal);
    gl_Position = scene.proj * scene.view * world;
}
