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
    vec4 uvScale;
    vec4 extra;
} modelData;

// Набор 2: материал (свой набор дескрипторов у каждого материала)
layout(set = 2, binding = 0) uniform sampler2D albedoMap;
layout(set = 2, binding = 1) uniform sampler2D specularMap;
layout(set = 2, binding = 2) uniform sampler2D emissiveMap;

// Управление нетривиальным сэмплированием - через push-константы
layout(push_constant) uniform PushConstants {
    uint flags;
    float time;
    float speed;
    float strength;
} pc;

const uint FLAG_ANIMATED_SAMPLING = 1u;

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec2 texCoord;
layout(location = 0) out vec4 outColor;

// Обычное сэмплирование либо нетривиальное: два слоя той же текстуры с
// разными координатами (сдвиг + волна, поворот + масштаб), зависящими от
// времени, и пульсирующая модуляция цвета.
vec3 sampleAlbedo(vec2 uv) {
    vec3 base = texture(albedoMap, uv).rgb;
    if ((pc.flags & FLAG_ANIMATED_SAMPLING) == 0u) {
        return base;
    }

    float t = pc.time * pc.speed;

    vec2 uvA = uv + vec2(0.10 * t, 0.05 * t)
                  + 0.02 * vec2(sin(uv.y * 12.0 + t * 2.0), cos(uv.x * 12.0 + t * 2.0));

    float angle = 0.35 * t;
    mat2 rotation = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
    vec2 uvB = rotation * (uv - 0.5) * 1.7 + 0.5 - vec2(0.07 * t, 0.0);

    vec3 layerA = texture(albedoMap, uvA).rgb;
    vec3 layerB = texture(albedoMap, uvB).rgb;
    vec3 blended = 0.5 * (layerA + layerB);

    vec3 pulse = 0.65 + 0.35 * (0.5 + 0.5 * vec3(sin(t * 1.3),
                                                 sin(t * 1.3 + 2.094),
                                                 sin(t * 1.3 + 4.188)));

    return mix(base, blended * pulse * 1.3, clamp(pc.strength, 0.0, 1.0));
}

vec3 blinnPhong(vec3 lightColor, float intensity, vec3 lightDirection,
                vec3 normal, vec3 viewDirection, vec3 albedo, vec3 specularColor) {
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

    vec3 diffuse = albedo * lightColor * diffuseFactor;
    vec3 specular = specularColor * lightColor * specularFactor;
    return intensity * (diffuse + specular);
}

void main() {
    vec3 normal = normalize(worldNormal);
    vec3 viewVector = scene.cameraPos.xyz - worldPosition;
    vec3 viewDirection = dot(viewVector, viewVector) > 1e-8
        ? normalize(viewVector)
        : normal;

    // Альбедо из текстуры; блеск - только там, где specular-текстура светлая
    vec3 albedo = sampleAlbedo(texCoord) * modelData.albedo.rgb;
    vec3 specularColor = modelData.specular.rgb * texture(specularMap, texCoord).rgb;

    vec3 result = scene.ambient.rgb * scene.ambient.a * albedo;

    // Направленный источник: направление хранит направление распространения света.
    if (scene.dirColor.a > 0.0) {
        vec3 lightDirection = normalize(-scene.dirDirection.xyz);
        result += blinnPhong(scene.dirColor.rgb, scene.dirColor.a, lightDirection,
                             normal, viewDirection, albedo, specularColor);
    }

    // Точечные источники: затухание по закону обратных квадратов.
    for (int i = 0; i < scene.counts.x; ++i) {
        vec3 toLight = pointLights[i].position.xyz - worldPosition;
        float distanceSquared = max(dot(toLight, toLight), 0.01);
        vec3 lightDirection = toLight * inversesqrt(distanceSquared);
        float attenuation = pointLights[i].color.a / distanceSquared;
        result += blinnPhong(pointLights[i].color.rgb, attenuation, lightDirection,
                             normal, viewDirection, albedo, specularColor);
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
        result += blinnPhong(spotLights[i].color.rgb, attenuation, lightDirection,
                             normal, viewDirection, albedo, specularColor);
    }

    // Излучение складывается с результатом Блинна-Фонга и минует расчёт освещения
    vec3 emissive = texture(emissiveMap, texCoord).rgb * modelData.extra.x;

    outColor = vec4(max(result, vec3(0.0)) + emissive, 1.0);
}
