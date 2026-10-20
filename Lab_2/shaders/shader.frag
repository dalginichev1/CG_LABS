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

struct PointLight {
    vec4 position;
    vec4 color;
};

layout(set = 0, binding = 1, std430) readonly buffer PointLightBuffer {
    PointLight pointLights[];
};

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color;
    vec4 cone;
};

layout(set = 0, binding = 2, std430) readonly buffer SpotLightBuffer {
    SpotLight spotLights[];
};

layout(set = 1, binding = 0, std140) uniform ModelData {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;
    vec4 specular;
} modelData;

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 0) out vec4 outColor;

vec3 blinnPhong(vec3 lightColor, float intensity, vec3 lightDirection,
                vec3 normal, vec3 viewDirection) {
    float diffuseFactor = max(dot(normal, lightDirection), 0.0);
    vec3 halfSum = lightDirection + viewDirection;
    vec3 halfDirection = dot(halfSum, halfSum) > 1e-8
        ? normalize(halfSum)
        : normal;

    float specularFactor = 0.0;
    if (diffuseFactor > 0.0) {
        specularFactor = pow(
            max(dot(normal, halfDirection), 0.0),
            max(modelData.specular.w, 1.0)
        );
    }

    vec3 diffuse = modelData.albedo.rgb * lightColor * diffuseFactor;
    vec3 specular = modelData.specular.rgb * lightColor * specularFactor;
    return intensity * (diffuse + specular);
}

void main() {
    vec3 normal = normalize(worldNormal);
    vec3 viewVector = scene.cameraPos.xyz - worldPosition;
    vec3 viewDirection = dot(viewVector, viewVector) > 1e-8
        ? normalize(viewVector)
        : normal;

    vec3 result = scene.ambient.rgb * scene.ambient.a * modelData.albedo.rgb;

    // Направленный источник: направление хранит направление распространения света.
    if (scene.dirColor.a > 0.0) {
        vec3 lightDirection = normalize(-scene.dirDirection.xyz);
        result += blinnPhong(scene.dirColor.rgb, scene.dirColor.a,
                             lightDirection, normal, viewDirection);
    }

    // Точечные источники: затухание по закону обратных квадратов.
    for (int i = 0; i < scene.counts.x; ++i) {
        vec3 toLight = pointLights[i].position.xyz - worldPosition;
        float distanceSquared = max(dot(toLight, toLight), 0.01);
        vec3 lightDirection = toLight * inversesqrt(distanceSquared);
        float attenuation = pointLights[i].color.a / distanceSquared;
        result += blinnPhong(pointLights[i].color.rgb, attenuation,
                             lightDirection, normal, viewDirection);
    }

    // Прожекторы: обратные квадраты и плавный переход между углами конуса.
    for (int i = 0; i < scene.counts.y; ++i) {
        vec3 fromLight = worldPosition - spotLights[i].position.xyz;
        float distanceSquared = max(dot(fromLight, fromLight), 0.01);
        vec3 lightToFragment = fromLight * inversesqrt(distanceSquared);
        vec3 lightDirection = -lightToFragment;

        float theta = dot(normalize(spotLights[i].direction.xyz), lightToFragment);
        float innerCos = spotLights[i].cone.x;
        float outerCos = spotLights[i].cone.y;
        float coneFactor;
        if (abs(innerCos - outerCos) < 1e-5) {
            coneFactor = step(outerCos, theta);
        } else {
            coneFactor = smoothstep(outerCos, innerCos, theta);
        }

        float attenuation = spotLights[i].color.a * coneFactor / distanceSquared;
        result += blinnPhong(spotLights[i].color.rgb, attenuation,
                             lightDirection, normal, viewDirection);
    }

    outColor = vec4(max(result, vec3(0.0)), modelData.albedo.a);
}
