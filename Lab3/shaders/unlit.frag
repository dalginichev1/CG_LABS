#version 450

layout(set = 1, binding = 0, std140) uniform ModelData {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;    // постоянный цвет фигуры визуализации
    vec4 specular;
    vec4 uvScale;
    vec4 extra;
} modelData;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = modelData.albedo;
}
