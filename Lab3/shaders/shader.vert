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

// Набор 1: динамический uniform buffer, смещение задаётся при привязке набора
layout(set = 1, binding = 0, std140) uniform ModelData {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;    // множитель (tint) для текстуры альбедо
    vec4 specular;  // rgb - цвет блика, w - коэффициент блика
    vec4 uvScale;   // xy - повторение текстуры по u и v
    vec4 extra;     // x - сила излучения (emissive)
} modelData;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec2 texCoord;

void main() {
    vec4 world = modelData.model * vec4(inPosition, 1.0);
    worldPosition = world.xyz;
    worldNormal = normalize(mat3(modelData.normalMatrix) * inNormal);
    texCoord = inUV * modelData.uvScale.xy;
    gl_Position = scene.proj * scene.view * world;
}
