#version 450
layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 baseColor;
} ubo;

void main() {
    outColor = vec4(fragColor * ubo.baseColor.rgb, 1.0);
}