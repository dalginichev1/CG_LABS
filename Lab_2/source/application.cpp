#include "application.hpp"

#include <imgui.h>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace application {
namespace {

// ---------------------------------------------------------------------------
// Данные, передаваемые в шейдеры (раскладка совпадает с std140 / std430)
// ---------------------------------------------------------------------------

struct Vertex {
  glm::vec3 pos;
  glm::vec3 normal;
};

// Набор дескрипторов 0 (сцена), binding 0: uniform buffer
struct SceneData {
  glm::mat4 view;
  glm::mat4 proj;
  glm::vec4 cameraPos;
  glm::vec4 ambient;      // rgb - цвет, w - интенсивность
  glm::vec4 dirDirection; // xyz - направление распространения света
  glm::vec4 dirColor;     // rgb - цвет, w - интенсивность
  glm::ivec4 counts;      // x - точечных, y - прожекторных
};

// Набор дескрипторов 0 (сцена), binding 1 и 2: storage buffers
struct PointLightGPU {
  glm::vec4 position;
  glm::vec4 color; // rgb - цвет, w - интенсивность
};

struct SpotLightGPU {
  glm::vec4 position;
  glm::vec4 direction;
  glm::vec4 color; // rgb - цвет, w - интенсивность
  glm::vec4 cone;  // x - cos(внутр. угол), y - cos(внешн. угол)
};

// Набор дескрипторов 1 (модель), binding 0: dynamic uniform buffer
struct ModelData {
  glm::mat4 model;
  glm::mat4 normalMatrix;
  glm::vec4 albedo;   // для фигур визуализации - постоянный цвет
  glm::vec4 specular; // rgb - цвет блика, w - коэффициент блика
};

static_assert(sizeof(SceneData) == 208, "SceneData must match std140 layout");
static_assert(sizeof(PointLightGPU) == 32, "PointLightGPU must match std430");
static_assert(sizeof(SpotLightGPU) == 64, "SpotLightGPU must match std430");
static_assert(sizeof(ModelData) == 160, "ModelData must match std140 layout");

constexpr uint32_t kMaxLights = 64; // ёмкость CPU-стороны для каждого массива
constexpr uint32_t kMaxDraws = 512; // 3 объекта + фигуры визуализации
constexpr int kConeSegments = 32;

// ---------------------------------------------------------------------------
// Состояние сцены (редактируется через ImGui)
// ---------------------------------------------------------------------------

struct Material {
  glm::vec3 albedo = {0.8f, 0.8f, 0.8f};
  glm::vec3 specular = {1.0f, 1.0f, 1.0f};
  float shininess = 32.0f;
};

struct ObjectState {
  glm::vec3 pos = {0.0f, 0.0f, 0.0f};
  glm::vec3 rot = {0.0f, 0.0f, 0.0f};
  glm::vec3 scale = {1.0f, 1.0f, 1.0f};
  Material material;

  bool animate = false;
  float anim_speed = 1.0f;
  float anim_radius = 2.0f;
  float anim_time = 0.0f;
};

struct Camera {
  glm::vec3 pos = {0.0f, 2.5f, 4.5f};
  float yaw = 0.0f;    // градусы, 0 = смотрим вдоль -Z, + = вправо
  float pitch = -29.0f; // градусы, + = вверх
  float fov = 45.0f;   // градусы, вертикальный
  float near_plane = 0.1f;
  float far_plane = 100.0f;
  bool perspective = true;
  float ortho_size = 4.0f;
  float move_speed = 3.0f;
};

struct DirectionalLight {
  bool enabled = true;
  glm::vec3 direction = {-0.4f, -1.0f, -0.3f};
  glm::vec3 color = {1.0f, 0.97f, 0.9f};
  float intensity = 0.5f;
};

struct PointLightState {
  glm::vec3 pos = {0.0f, 1.0f, 1.5f};
  glm::vec3 color = {1.0f, 0.6f, 0.3f};
  float intensity = 8.0f;

  bool animate = false;
  float anim_speed = 1.0f;
  float anim_radius = 2.5f;
  float anim_time = 0.0f;
};

struct SpotLightState {
  glm::vec3 pos = {2.5f, 3.0f, 2.0f};
  glm::vec3 direction = {-0.5f, -0.6f, -0.4f};
  glm::vec3 color = {0.4f, 0.6f, 1.0f};
  float intensity = 60.0f;
  float inner_angle = 15.0f; // градусы
  float outer_angle = 25.0f; // градусы

  bool animate = false;
  float anim_speed = 0.7f;
  float anim_radius = 3.0f;
  float anim_time = 0.0f;
};

constexpr glm::vec3 kAimTarget = {0.0f, -0.55f, 0.0f};

Camera camera;
glm::vec3 ambient_color = {1.0f, 1.0f, 1.0f};
float ambient_intensity = 0.08f;
DirectionalLight dir_light;
std::vector<PointLightState> point_lights;
std::vector<SpotLightState> spot_lights;

bool show_light_gizmos = true;
bool show_spot_cones = true;

// 0 - пол, 1 - первый куб, 2 - второй куб
ObjectState objects[3];

double last_time = 0.0;

// ---------------------------------------------------------------------------
// Ресурсы Vulkan
// ---------------------------------------------------------------------------

struct GpuBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation allocation = nullptr;
  void *mapped = nullptr;
};

VkDescriptorSetLayout sceneSetLayout;
VkDescriptorSetLayout modelSetLayout;
VkDescriptorPool descriptorPool;
VkDescriptorSet sceneSet;
VkDescriptorSet modelSet;

VkPipelineLayout pipelineLayout;
VkPipeline litPipeline;        // освещение по Блинну-Фонгу, треугольники
VkPipeline unlitSolidPipeline; // постоянный цвет, треугольники
VkPipeline unlitLinePipeline;  // постоянный цвет, линии

GpuBuffer cubeVertexBuffer;
GpuBuffer cubeIndexBuffer;
GpuBuffer coneVertexBuffer;
uint32_t cubeIndexCount = 0;
uint32_t coneVertexCount = 0;

GpuBuffer sceneUbo;
GpuBuffer pointLightsSsbo;
GpuBuffer spotLightsSsbo;
GpuBuffer modelUbo;
VkDeviceSize modelStride = 0; // sizeof(ModelData) с учётом выравнивания

enum class DrawKind { Lit, UnlitSolid, UnlitLines };

struct DrawCall {
  DrawKind kind;
  uint32_t dynamicOffset;
};

std::vector<DrawCall> draw_calls;

// ---------------------------------------------------------------------------
// Процедурная генерация геометрии
// ---------------------------------------------------------------------------

// Куб 1x1x1: 6 граней по 4 вершины. Нормаль каждой грани вычисляется
// векторным произведением рёбер первого треугольника грани.
void buildCube(std::vector<Vertex> &vertices, std::vector<uint16_t> &indices) {
  struct Face {
    glm::vec3 center; // центр грани
    glm::vec3 u;      // направление "вправо" по грани
    glm::vec3 v;      // направление "вверх" по грани
  };

  // Для каждой грани cross(u, v) направлен наружу куба.
  const Face faces[6] = {
      {{0.5f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}},  // +X
      {{-0.5f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},  // -X
      {{0.0f, 0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}},  // +Y
      {{0.0f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},  // -Y
      {{0.0f, 0.0f, 0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},   // +Z
      {{0.0f, 0.0f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, // -Z
  };
  const float corners[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};

  for (const Face &f : faces) {
    const uint16_t base = static_cast<uint16_t>(vertices.size());

    glm::vec3 p[4];
    for (int i = 0; i < 4; i++) {
      p[i] = f.center + f.u * (0.5f * corners[i][0]) +
             f.v * (0.5f * corners[i][1]);
    }

    const glm::vec3 normal = glm::normalize(glm::cross(p[1] - p[0], p[2] - p[0]));
    for (int i = 0; i < 4; i++) {
      vertices.push_back({p[i], normal});
    }

    const uint16_t quad[6] = {0, 1, 2, 2, 3, 0};
    for (uint16_t q : quad) {
      indices.push_back(static_cast<uint16_t>(base + q));
    }
  }
}

// Конус единичной длины для визуализации прожектора: вершина в начале
// координат, ось направлена вдоль -Z, основание - окружность радиуса 1 в z=-1.
// Список линий (LINE_LIST). Нормали не используются.
void buildConeLines(std::vector<Vertex> &vertices) {
  const float two_pi = 6.28318530718f;
  auto add_line = [&](glm::vec3 a, glm::vec3 b) {
    vertices.push_back({a, {0.0f, 0.0f, 0.0f}});
    vertices.push_back({b, {0.0f, 0.0f, 0.0f}});
  };
  auto ring_point = [&](int i) {
    const float a = two_pi * static_cast<float>(i) / kConeSegments;
    return glm::vec3(std::cos(a), std::sin(a), -1.0f);
  };

  for (int i = 0; i < kConeSegments; i++) {
    add_line(ring_point(i), ring_point(i + 1)); // окружность основания
  }
  for (int i = 0; i < kConeSegments; i += kConeSegments / 8) {
    add_line({0.0f, 0.0f, 0.0f}, ring_point(i)); // образующие
  }
  add_line({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}); // ось
}

// ---------------------------------------------------------------------------
// Вспомогательные функции Vulkan
// ---------------------------------------------------------------------------

std::vector<char> readFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("Не удалось открыть файл шейдера: " + filename);
  size_t fileSize = static_cast<size_t>(file.tellg());
  std::vector<char> buffer(fileSize);
  file.seekg(0);
  file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
  file.close();
  return buffer;
}

VkShaderModule createShaderModule(const std::vector<char> &code) {
  auto &ctx = graphics::internal::context;
  VkShaderModuleCreateInfo createInfo{
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  createInfo.codeSize = code.size();
  createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());
  VkShaderModule shaderModule;
  if (vkCreateShaderModule(ctx.device, &createInfo, nullptr, &shaderModule) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create shader module!");
  return shaderModule;
}

GpuBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
  auto &ctx = graphics::internal::context;
  VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo vmaAllocInfo{};
  vmaAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
  vmaAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT;
  vmaAllocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  GpuBuffer result;
  VmaAllocationInfo info{};
  VkResult vr = vmaCreateBuffer(ctx.allocator, &bufferInfo, &vmaAllocInfo,
                                &result.buffer, &result.allocation, &info);
  if (vr != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan buffer! VkResult = " +
                             std::to_string(vr));
  }
  result.mapped = info.pMappedData;
  return result;
}

void destroyBuffer(GpuBuffer &buf) {
  auto &ctx = graphics::internal::context;
  vmaDestroyBuffer(ctx.allocator, buf.buffer, buf.allocation);
  buf = {};
}

VkDescriptorSetLayout createSetLayout(
    const std::vector<VkDescriptorSetLayoutBinding> &bindings) {
  auto &ctx = graphics::internal::context;
  VkDescriptorSetLayoutCreateInfo info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  info.bindingCount = static_cast<uint32_t>(bindings.size());
  info.pBindings = bindings.data();
  VkDescriptorSetLayout layout;
  if (vkCreateDescriptorSetLayout(ctx.device, &info, nullptr, &layout) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create descriptor set layout!");
  }
  return layout;
}

VkWriteDescriptorSet makeWrite(VkDescriptorSet set, uint32_t binding,
                               VkDescriptorType type,
                               const VkDescriptorBufferInfo *info) {
  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = set;
  write.dstBinding = binding;
  write.dstArrayElement = 0;
  write.descriptorType = type;
  write.descriptorCount = 1;
  write.pBufferInfo = info;
  return write;
}

VkPipeline createPipeline(VkShaderModule vert, VkShaderModule frag,
                          VkPrimitiveTopology topology) {
  auto &ctx = graphics::internal::context;

  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vert;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = frag;
  stages[1].pName = "main";

  VkVertexInputBindingDescription binding{};
  binding.binding = 0;
  binding.stride = sizeof(Vertex);
  binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  VkVertexInputAttributeDescription attributes[2]{};
  attributes[0].binding = 0;
  attributes[0].location = 0;
  attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributes[0].offset = offsetof(Vertex, pos);
  attributes[1].binding = 0;
  attributes[1].location = 1;
  attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributes[1].offset = offsetof(Vertex, normal);

  VkPipelineVertexInputStateCreateInfo vertexInput{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vertexInput.vertexBindingDescriptionCount = 1;
  vertexInput.pVertexBindingDescriptions = &binding;
  vertexInput.vertexAttributeDescriptionCount = 2;
  vertexInput.pVertexAttributeDescriptions = attributes;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  inputAssembly.topology = topology;
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  VkPipelineViewportStateCreateInfo viewportState{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo rasterizer{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_NONE;
  rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisampling{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo depthStencil{
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  depthStencil.depthTestEnable = VK_TRUE;
  depthStencil.depthWriteEnable = VK_TRUE;
  depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

  VkPipelineColorBlendAttachmentState colorBlendAttachment{};
  colorBlendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo colorBlending{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;

  const VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                          VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamicState{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamicState.dynamicStateCount = 2;
  dynamicState.pDynamicStates = dynamicStates;

  VkGraphicsPipelineCreateInfo pipelineInfo{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = stages;
  pipelineInfo.pVertexInputState = &vertexInput;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pDepthStencilState = &depthStencil;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = &dynamicState;
  pipelineInfo.layout = pipelineLayout;
  pipelineInfo.renderPass = ctx.render_pass;
  pipelineInfo.subpass = 0;

  VkPipeline pipeline = VK_NULL_HANDLE;
  VkResult result = vkCreateGraphicsPipelines(
      ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create graphics pipeline! VkResult = " +
                             std::to_string(result));
  }
  return pipeline;
}

// ---------------------------------------------------------------------------
// Математика сцены
// ---------------------------------------------------------------------------

glm::mat4 objectModelMatrix(const ObjectState &obj) {
  glm::mat4 model = glm::translate(glm::mat4(1.0f), obj.pos);
  model = glm::rotate(model, obj.rot.x, glm::vec3(1.0f, 0.0f, 0.0f));
  model = glm::rotate(model, obj.rot.y, glm::vec3(0.0f, 1.0f, 0.0f));
  model = glm::rotate(model, obj.rot.z, glm::vec3(0.0f, 0.0f, 1.0f));
  model = glm::scale(model, obj.scale);
  return model;
}

glm::vec3 safeNormalize(const glm::vec3 &v, const glm::vec3 &fallback) {
  const float len = glm::length(v);
  return len > 1e-5f ? v / len : fallback;
}

glm::vec3 cameraForward(const Camera &c) {
  const float y = glm::radians(c.yaw);
  const float p = glm::radians(c.pitch);
  return glm::normalize(glm::vec3(std::sin(y) * std::cos(p), std::sin(p),
                                  -std::cos(y) * std::cos(p)));
}

// Матрица конуса: вершина в pos, ось вдоль dir, длина length,
// угол при вершине (полу-угол) angle_rad.
glm::mat4 coneModelMatrix(const glm::vec3 &pos, const glm::vec3 &dir,
                          float angle_rad, float length) {
  const glm::vec3 up = std::fabs(dir.y) > 0.99f ? glm::vec3(1.0f, 0.0f, 0.0f)
                                                : glm::vec3(0.0f, 1.0f, 0.0f);
  // lookAt переводит направление dir в -Z, обратная матрица - наоборот.
  const glm::mat4 orient = glm::inverse(glm::lookAt(pos, pos + dir, up));
  const float r = std::tan(angle_rad) * length;
  return orient * glm::scale(glm::mat4(1.0f), glm::vec3(r, r, length));
}

// ---------------------------------------------------------------------------
// Анимация и ввод
// ---------------------------------------------------------------------------

void updateObjectAnimation(ObjectState &obj, float dt) {
  if (!obj.animate)
    return;
  obj.anim_time += dt * obj.anim_speed;
  obj.pos.x = std::cos(obj.anim_time) * obj.anim_radius;
  obj.pos.z = std::sin(obj.anim_time) * obj.anim_radius;
  obj.rot.y += dt * obj.anim_speed;
  obj.rot.x += dt * (obj.anim_speed * 0.5f);
}

void updateLightAnimation(PointLightState &l, float dt) {
  if (!l.animate)
    return;
  l.anim_time += dt * l.anim_speed;
  l.pos.x = std::cos(l.anim_time) * l.anim_radius;
  l.pos.z = std::sin(l.anim_time) * l.anim_radius;
}

void updateLightAnimation(SpotLightState &l, float dt) {
  if (!l.animate)
    return;
  l.anim_time += dt * l.anim_speed;
  l.pos.x = std::cos(l.anim_time) * l.anim_radius;
  l.pos.z = std::sin(l.anim_time) * l.anim_radius;
  // Во время движения прожектор следит за центром сцены
  l.direction = safeNormalize(kAimTarget - l.pos, {0.0f, -1.0f, 0.0f});
}

// Управление камерой с клавиатуры/мыши: WASD + QE, ПКМ - вращение
void updateCameraInput(float dt) {
  ImGuiIO &io = ImGui::GetIO();

  if (!io.WantCaptureMouse && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
    camera.yaw += io.MouseDelta.x * 0.15f;
    camera.pitch -= io.MouseDelta.y * 0.15f;
  }

  if (!io.WantCaptureKeyboard) {
    const glm::vec3 forward = cameraForward(camera);
    const glm::vec3 right =
        glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
    float speed = camera.move_speed * dt;
    if (ImGui::IsKeyDown(ImGuiKey_LeftShift))
      speed *= 3.0f;

    if (ImGui::IsKeyDown(ImGuiKey_W))
      camera.pos += forward * speed;
    if (ImGui::IsKeyDown(ImGuiKey_S))
      camera.pos -= forward * speed;
    if (ImGui::IsKeyDown(ImGuiKey_D))
      camera.pos += right * speed;
    if (ImGui::IsKeyDown(ImGuiKey_A))
      camera.pos -= right * speed;
    if (ImGui::IsKeyDown(ImGuiKey_E))
      camera.pos.y += speed;
    if (ImGui::IsKeyDown(ImGuiKey_Q))
      camera.pos.y -= speed;
  }

  camera.pitch = std::clamp(camera.pitch, -89.0f, 89.0f);
  if (camera.yaw > 180.0f)
    camera.yaw -= 360.0f;
  if (camera.yaw < -180.0f)
    camera.yaw += 360.0f;
}

// ---------------------------------------------------------------------------
// Пользовательский интерфейс
// ---------------------------------------------------------------------------

void drawCameraUI() {
  ImGui::DragFloat3("Позиция", glm::value_ptr(camera.pos), 0.05f);
  ImGui::SliderFloat("Рыскание (yaw)", &camera.yaw, -180.0f, 180.0f, "%.1f°");
  ImGui::SliderFloat("Тангаж (pitch)", &camera.pitch, -89.0f, 89.0f, "%.1f°");
  ImGui::SliderFloat("Угол обзора (FOV)", &camera.fov, 10.0f, 120.0f, "%.1f°");
  ImGui::SliderFloat("Скорость движения", &camera.move_speed, 0.5f, 20.0f);

  if (ImGui::RadioButton("Перспективная", camera.perspective))
    camera.perspective = true;
  ImGui::SameLine();
  if (ImGui::RadioButton("Ортографическая", !camera.perspective))
    camera.perspective = false;
  if (!camera.perspective)
    ImGui::SliderFloat("Размер ортографической проекции", &camera.ortho_size,
                       0.5f, 20.0f);

  ImGui::TextDisabled("WASD/QE - перемещение, Shift - ускорение,");
  ImGui::TextDisabled("зажатая ПКМ + мышь - поворот камеры");
  if (ImGui::Button("Сбросить камеру"))
    camera = Camera{};
}

void drawMaterialUI(Material &m) {
  ImGui::ColorEdit3("Альбедо", glm::value_ptr(m.albedo));
  ImGui::ColorEdit3("Цвет блика", glm::value_ptr(m.specular));
  ImGui::SliderFloat("Коэффициент блика", &m.shininess, 1.0f, 512.0f, "%.1f",
                     ImGuiSliderFlags_Logarithmic);
}

void drawObjectUI(const char *title, const char *id, ObjectState &obj,
                  bool animatable) {
  ImGui::PushID(id);
  if (ImGui::CollapsingHeader(title)) {
    ImGui::Text("Трансформации");
    ImGui::DragFloat3("Позиция", glm::value_ptr(obj.pos), 0.05f);
    ImGui::DragFloat3("Поворот", glm::value_ptr(obj.rot), 0.05f);
    ImGui::DragFloat3("Масштаб", glm::value_ptr(obj.scale), 0.05f);

    if (animatable) {
      ImGui::Text("Анимация");
      ImGui::Checkbox("Движение по кругу", &obj.animate);
      ImGui::SliderFloat("Скорость", &obj.anim_speed, 0.1f, 5.0f);
      ImGui::SliderFloat("Радиус траектории", &obj.anim_radius, 0.0f, 10.0f);
    }

    ImGui::Text("Материал (Блинн-Фонг)");
    drawMaterialUI(obj.material);
  }
  ImGui::PopID();
}

void drawDirectionalLightUI() {
  ImGui::Checkbox("Включён", &dir_light.enabled);
  ImGui::DragFloat3("Направление света", glm::value_ptr(dir_light.direction),
                    0.01f, -1.0f, 1.0f);
  ImGui::ColorEdit3("Цвет", glm::value_ptr(dir_light.color));
  ImGui::SliderFloat("Интенсивность", &dir_light.intensity, 0.0f, 5.0f);
}

void drawPointLightsUI() {
  int remove_index = -1;
  for (size_t i = 0; i < point_lights.size(); i++) {
    PointLightState &l = point_lights[i];
    ImGui::PushID(static_cast<int>(i));
    const std::string title = "Точечный источник #" + std::to_string(i + 1);
    if (ImGui::CollapsingHeader(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::DragFloat3("Позиция", glm::value_ptr(l.pos), 0.05f);
      ImGui::ColorEdit3("Цвет", glm::value_ptr(l.color));
      ImGui::SliderFloat("Интенсивность", &l.intensity, 0.0f, 100.0f);
      ImGui::Checkbox("Движение по кругу", &l.animate);
      if (l.animate) {
        ImGui::SliderFloat("Скорость", &l.anim_speed, 0.1f, 5.0f);
        ImGui::SliderFloat("Радиус", &l.anim_radius, 0.0f, 10.0f);
      }
      if (ImGui::Button("Удалить"))
        remove_index = static_cast<int>(i);
    }
    ImGui::PopID();
  }
  if (remove_index >= 0)
    point_lights.erase(point_lights.begin() + remove_index);

  if (ImGui::Button("Добавить точечный источник") &&
      point_lights.size() < kMaxLights) {
    point_lights.push_back(PointLightState{});
  }
}

void drawSpotLightsUI() {
  int remove_index = -1;
  for (size_t i = 0; i < spot_lights.size(); i++) {
    SpotLightState &l = spot_lights[i];
    ImGui::PushID(static_cast<int>(i));
    const std::string title = "Прожектор #" + std::to_string(i + 1);
    if (ImGui::CollapsingHeader(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::DragFloat3("Позиция", glm::value_ptr(l.pos), 0.05f);
      ImGui::DragFloat3("Направление", glm::value_ptr(l.direction), 0.01f);
      if (ImGui::Button("Направить в центр сцены"))
        l.direction = safeNormalize(kAimTarget - l.pos, {0.0f, -1.0f, 0.0f});
      ImGui::ColorEdit3("Цвет", glm::value_ptr(l.color));
      ImGui::SliderFloat("Интенсивность", &l.intensity, 0.0f, 500.0f);

      ImGui::SliderFloat("Внутренний угол", &l.inner_angle, 0.0f, 80.0f,
                         "%.1f°");
      ImGui::SliderFloat("Внешний угол", &l.outer_angle, 0.0f, 80.0f, "%.1f°");
      // Внутренний угол не может превышать внешний
      l.inner_angle = std::min(l.inner_angle, l.outer_angle);

      ImGui::Checkbox("Движение по кругу (следит за центром)", &l.animate);
      if (l.animate) {
        ImGui::SliderFloat("Скорость", &l.anim_speed, 0.1f, 5.0f);
        ImGui::SliderFloat("Радиус", &l.anim_radius, 0.0f, 10.0f);
      }
      if (ImGui::Button("Удалить"))
        remove_index = static_cast<int>(i);
    }
    ImGui::PopID();
  }
  if (remove_index >= 0)
    spot_lights.erase(spot_lights.begin() + remove_index);

  if (ImGui::Button("Добавить прожектор") && spot_lights.size() < kMaxLights) {
    SpotLightState l;
    l.direction = safeNormalize(kAimTarget - l.pos, {0.0f, -1.0f, 0.0f});
    spot_lights.push_back(l);
  }
}

void drawUI() {
  ImGui::SetNextWindowSize(ImVec2(430, 680), ImGuiCond_FirstUseEver);
  ImGui::Begin("Настройки 3D сцены");

  if (ImGui::CollapsingHeader("Камера", ImGuiTreeNodeFlags_DefaultOpen))
    drawCameraUI();

  if (ImGui::CollapsingHeader("Освещение", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Text("Рассеянный свет (ambient)");
    ImGui::ColorEdit3("Цвет##ambient", glm::value_ptr(ambient_color));
    ImGui::SliderFloat("Интенсивность##ambient", &ambient_intensity, 0.0f, 1.0f);
    ImGui::Separator();

    ImGui::Text("Направленный источник");
    ImGui::PushID("dir");
    drawDirectionalLightUI();
    ImGui::PopID();
    ImGui::Separator();

    ImGui::Text("Визуализация источников (Доп. 4)");
    ImGui::Checkbox("Показывать источники", &show_light_gizmos);
    ImGui::Checkbox("Показывать конусы прожекторов", &show_spot_cones);
  }

  if (ImGui::CollapsingHeader("Точечные источники (Доп. 2)"))
    drawPointLightsUI();

  if (ImGui::CollapsingHeader("Прожекторы (Доп. 3)"))
    drawSpotLightsUI();

  if (ImGui::CollapsingHeader("Объекты", ImGuiTreeNodeFlags_DefaultOpen)) {
    drawObjectUI("Пол", "floor", objects[0], false);
    drawObjectUI("Первый куб", "cube1", objects[1], true);
    drawObjectUI("Второй куб", "cube2", objects[2], true);
  }

  ImGui::End();
}

void initializeScene() {
  objects[0].pos = {0.0f, -1.2f, 0.0f};
  objects[0].scale = {12.0f, 0.3f, 12.0f};
  objects[0].material.albedo = {0.55f, 0.55f, 0.58f};
  objects[0].material.specular = {0.3f, 0.3f, 0.3f};
  objects[0].material.shininess = 16.0f;

  objects[1].pos = {-1.0f, -0.55f, 0.0f};
  objects[1].material.albedo = {0.85f, 0.35f, 0.3f};
  objects[1].material.shininess = 64.0f;

  objects[2].pos = {1.5f, -0.55f, -0.5f};
  objects[2].material.albedo = {0.3f, 0.7f, 0.4f};
  objects[2].material.shininess = 128.0f;

  point_lights.push_back(PointLightState{});

  SpotLightState spot;
  spot.direction = safeNormalize(kAimTarget - spot.pos, {0.0f, -1.0f, 0.0f});
  spot_lights.push_back(spot);
}

} // namespace

// ---------------------------------------------------------------------------
// Жизненный цикл приложения
// ---------------------------------------------------------------------------

bool initialize() {
  auto &ctx = graphics::internal::context;

  initializeScene();

  // --- Геометрия -----------------------------------------------------------
  std::vector<Vertex> cubeVertices;
  std::vector<uint16_t> cubeIndices;
  buildCube(cubeVertices, cubeIndices);
  cubeIndexCount = static_cast<uint32_t>(cubeIndices.size());

  std::vector<Vertex> coneVertices;
  buildConeLines(coneVertices);
  coneVertexCount = static_cast<uint32_t>(coneVertices.size());

  cubeVertexBuffer = createBuffer(sizeof(Vertex) * cubeVertices.size(),
                                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  std::memcpy(cubeVertexBuffer.mapped, cubeVertices.data(),
              sizeof(Vertex) * cubeVertices.size());

  cubeIndexBuffer = createBuffer(sizeof(uint16_t) * cubeIndices.size(),
                                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
  std::memcpy(cubeIndexBuffer.mapped, cubeIndices.data(),
              sizeof(uint16_t) * cubeIndices.size());

  coneVertexBuffer = createBuffer(sizeof(Vertex) * coneVertices.size(),
                                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  std::memcpy(coneVertexBuffer.mapped, coneVertices.data(),
              sizeof(Vertex) * coneVertices.size());

  // --- Буферы данных -------------------------------------------------------
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(ctx.physical_device, &props);
  const VkDeviceSize align =
      std::max<VkDeviceSize>(props.limits.minUniformBufferOffsetAlignment, 1);
  modelStride = (sizeof(ModelData) + align - 1) / align * align;

  sceneUbo = createBuffer(sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  pointLightsSsbo = createBuffer(sizeof(PointLightGPU) * kMaxLights,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  spotLightsSsbo = createBuffer(sizeof(SpotLightGPU) * kMaxLights,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  modelUbo = createBuffer(modelStride * kMaxDraws,
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

  // --- Наборы дескрипторов -------------------------------------------------
  // Набор 0 (сцена): UBO + 2 storage buffer
  {
    std::vector<VkDescriptorSetLayoutBinding> b(3);
    b[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT,
            nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT,
            nullptr};
    sceneSetLayout = createSetLayout(b);
  }
  // Набор 1 (модель): dynamic UBO (трансформация + материал)
  {
    std::vector<VkDescriptorSetLayoutBinding> b(1);
    b[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    modelSetLayout = createSetLayout(b);
  }

  const VkDescriptorPoolSize poolSizes[] = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1},
  };
  VkDescriptorPoolCreateInfo poolInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolInfo.poolSizeCount = 3;
  poolInfo.pPoolSizes = poolSizes;
  poolInfo.maxSets = 2;
  if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create descriptor pool!");
  }

  {
    VkDescriptorSetAllocateInfo allocInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;

    allocInfo.pSetLayouts = &sceneSetLayout;
    if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &sceneSet) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to allocate scene descriptor set!");

    allocInfo.pSetLayouts = &modelSetLayout;
    if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &modelSet) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to allocate model descriptor set!");
  }

  {
    VkDescriptorBufferInfo sceneInfo{sceneUbo.buffer, 0, sizeof(SceneData)};
    VkDescriptorBufferInfo pointInfo{pointLightsSsbo.buffer, 0,
                                     sizeof(PointLightGPU) * kMaxLights};
    VkDescriptorBufferInfo spotInfo{spotLightsSsbo.buffer, 0,
                                    sizeof(SpotLightGPU) * kMaxLights};
    VkDescriptorBufferInfo modelInfo{modelUbo.buffer, 0, sizeof(ModelData)};

    VkWriteDescriptorSet writes[] = {
        makeWrite(sceneSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &sceneInfo),
        makeWrite(sceneSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &pointInfo),
        makeWrite(sceneSet, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &spotInfo),
        makeWrite(modelSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                  &modelInfo),
    };
    vkUpdateDescriptorSets(ctx.device, 4, writes, 0, nullptr);
  }

  // --- Конвейеры -----------------------------------------------------------
  const VkDescriptorSetLayout setLayouts[] = {sceneSetLayout, modelSetLayout};
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutInfo.setLayoutCount = 2;
  pipelineLayoutInfo.pSetLayouts = setLayouts;
  if (vkCreatePipelineLayout(ctx.device, &pipelineLayoutInfo, nullptr,
                             &pipelineLayout) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create pipeline layout!");
  }

  VkShaderModule litVert = createShaderModule(readFile("shaders/shader.vert.spv"));
  VkShaderModule litFrag = createShaderModule(readFile("shaders/shader.frag.spv"));
  VkShaderModule unlitVert = createShaderModule(readFile("shaders/unlit.vert.spv"));
  VkShaderModule unlitFrag = createShaderModule(readFile("shaders/unlit.frag.spv"));

  litPipeline =
      createPipeline(litVert, litFrag, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  unlitSolidPipeline =
      createPipeline(unlitVert, unlitFrag, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  unlitLinePipeline =
      createPipeline(unlitVert, unlitFrag, VK_PRIMITIVE_TOPOLOGY_LINE_LIST);

  vkDestroyShaderModule(ctx.device, unlitFrag, nullptr);
  vkDestroyShaderModule(ctx.device, unlitVert, nullptr);
  vkDestroyShaderModule(ctx.device, litFrag, nullptr);
  vkDestroyShaderModule(ctx.device, litVert, nullptr);

  return true;
}

void shutdown() {
  auto &ctx = graphics::internal::context;
  vkDeviceWaitIdle(ctx.device);

  vkDestroyPipeline(ctx.device, litPipeline, nullptr);
  vkDestroyPipeline(ctx.device, unlitSolidPipeline, nullptr);
  vkDestroyPipeline(ctx.device, unlitLinePipeline, nullptr);
  vkDestroyPipelineLayout(ctx.device, pipelineLayout, nullptr);
  vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(ctx.device, sceneSetLayout, nullptr);
  vkDestroyDescriptorSetLayout(ctx.device, modelSetLayout, nullptr);

  destroyBuffer(cubeVertexBuffer);
  destroyBuffer(cubeIndexBuffer);
  destroyBuffer(coneVertexBuffer);
  destroyBuffer(sceneUbo);
  destroyBuffer(pointLightsSsbo);
  destroyBuffer(spotLightsSsbo);
  destroyBuffer(modelUbo);
}

void update(double time) {
  if (last_time == 0.0)
    last_time = time;
  const float dt = static_cast<float>(time - last_time);
  last_time = time;

  updateCameraInput(dt);
  drawUI();

  updateObjectAnimation(objects[1], dt);
  updateObjectAnimation(objects[2], dt);
  for (auto &l : point_lights)
    updateLightAnimation(l, dt);
  for (auto &l : spot_lights)
    updateLightAnimation(l, dt);

  auto &ctx = graphics::internal::context;
  const float aspect = static_cast<float>(ctx.swapchain_extent.width) /
                       static_cast<float>(std::max(ctx.swapchain_extent.height, 1u));

  // --- Камера --------------------------------------------------------------
  const glm::vec3 forward = cameraForward(camera);
  const glm::mat4 view = glm::lookAt(camera.pos, camera.pos + forward,
                                     glm::vec3(0.0f, 1.0f, 0.0f));

  glm::mat4 proj;
  if (camera.perspective) {
    proj = glm::perspective(glm::radians(camera.fov), aspect,
                            camera.near_plane, camera.far_plane);
  } else {
    const float s = camera.ortho_size;
    proj = glm::ortho(-aspect * s, aspect * s, -s, s, camera.near_plane,
                      camera.far_plane);
  }
  proj[1][1] *= -1.0f; // ось Y в Vulkan направлена вниз

  // --- Данные сцены (набор дескрипторов 0) ----------------------------------
  SceneData scene{};
  scene.view = view;
  scene.proj = proj;
  scene.cameraPos = glm::vec4(camera.pos, 1.0f);
  scene.ambient = glm::vec4(ambient_color, ambient_intensity);
  scene.dirDirection = glm::vec4(
      safeNormalize(dir_light.direction, {0.0f, -1.0f, 0.0f}), 0.0f);
  scene.dirColor =
      glm::vec4(dir_light.color, dir_light.enabled ? dir_light.intensity : 0.0f);
  scene.counts = glm::ivec4(static_cast<int>(point_lights.size()),
                            static_cast<int>(spot_lights.size()), 0, 0);
  std::memcpy(sceneUbo.mapped, &scene, sizeof(scene));

  // Массивы источников в storage buffers
  {
    auto *dst = static_cast<PointLightGPU *>(pointLightsSsbo.mapped);
    for (size_t i = 0; i < point_lights.size(); i++) {
      const PointLightState &l = point_lights[i];
      dst[i].position = glm::vec4(l.pos, 1.0f);
      dst[i].color = glm::vec4(l.color, l.intensity);
    }
  }
  {
    auto *dst = static_cast<SpotLightGPU *>(spotLightsSsbo.mapped);
    for (size_t i = 0; i < spot_lights.size(); i++) {
      const SpotLightState &l = spot_lights[i];
      const float outer = glm::radians(l.outer_angle);
      const float inner = glm::radians(std::min(l.inner_angle, l.outer_angle));
      dst[i].position = glm::vec4(l.pos, 1.0f);
      dst[i].direction = glm::vec4(
          safeNormalize(l.direction, {0.0f, -1.0f, 0.0f}), 0.0f);
      dst[i].color = glm::vec4(l.color, l.intensity);
      dst[i].cone = glm::vec4(std::cos(inner), std::cos(outer), 0.0f, 0.0f);
    }
  }

  // --- Данные моделей (набор дескрипторов 1) и список отрисовки --------------
  draw_calls.clear();
  auto *modelBytes = static_cast<uint8_t *>(modelUbo.mapped);

  auto pushDraw = [&](DrawKind kind, const glm::mat4 &model,
                      const glm::vec4 &albedo, const glm::vec4 &specular) {
    if (draw_calls.size() >= kMaxDraws)
      return;
    const uint32_t offset =
        static_cast<uint32_t>(draw_calls.size() * modelStride);

    ModelData md{};
    md.model = model;
    md.normalMatrix = glm::mat4(glm::transpose(glm::inverse(glm::mat3(model))));
    md.albedo = albedo;
    md.specular = specular;
    std::memcpy(modelBytes + offset, &md, sizeof(md));

    draw_calls.push_back({kind, offset});
  };

  // Объекты сцены (освещаются по Блинну-Фонгу)
  for (const ObjectState &obj : objects) {
    pushDraw(DrawKind::Lit, objectModelMatrix(obj),
             glm::vec4(obj.material.albedo, 1.0f),
             glm::vec4(obj.material.specular, obj.material.shininess));
  }

  // Визуализация источников (постоянный цвет, без освещения)
  if (show_light_gizmos) {
    const glm::vec4 unused(0.0f);

    for (const PointLightState &l : point_lights) {
      glm::mat4 m = glm::translate(glm::mat4(1.0f), l.pos);
      m = glm::scale(m, glm::vec3(0.15f));
      pushDraw(DrawKind::UnlitSolid, m,
               glm::vec4(glm::clamp(l.color, 0.0f, 1.0f), 1.0f), unused);
    }

    for (const SpotLightState &l : spot_lights) {
      glm::mat4 m = glm::translate(glm::mat4(1.0f), l.pos);
      m = glm::scale(m, glm::vec3(0.15f));
      pushDraw(DrawKind::UnlitSolid, m,
               glm::vec4(glm::clamp(l.color, 0.0f, 1.0f), 1.0f), unused);

      if (show_spot_cones) {
        const glm::vec3 dir = safeNormalize(l.direction, {0.0f, -1.0f, 0.0f});
        const float outer = glm::radians(l.outer_angle);
        const float inner = glm::radians(std::min(l.inner_angle, l.outer_angle));
        const float length = 2.0f;

        // Внешний конус - белый, внутренний - серый
        pushDraw(DrawKind::UnlitLines, coneModelMatrix(l.pos, dir, outer, length),
                 glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), unused);
        pushDraw(DrawKind::UnlitLines, coneModelMatrix(l.pos, dir, inner, length),
                 glm::vec4(0.55f, 0.55f, 0.55f, 1.0f), unused);
      }
    }
  }
}

void render(const graphics::internal::FrameData &fd) {
  auto &ctx = graphics::internal::context;

  VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(fd.command_buffer, &beginInfo);

  VkClearValue clearValues[2] = {};
  clearValues[0].color = {{0.1f, 0.1f, 0.15f, 1.0f}};
  clearValues[1].depthStencil = {1.0f, 0};

  VkRenderPassBeginInfo renderPassInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  renderPassInfo.renderPass = ctx.render_pass;
  renderPassInfo.framebuffer = fd.framebuffer;
  renderPassInfo.renderArea.offset = {0, 0};
  renderPassInfo.renderArea.extent = ctx.swapchain_extent;
  renderPassInfo.clearValueCount = 2;
  renderPassInfo.pClearValues = clearValues;

  vkCmdBeginRenderPass(fd.command_buffer, &renderPassInfo,
                       VK_SUBPASS_CONTENTS_INLINE);

  VkViewport viewport{};
  viewport.width = static_cast<float>(ctx.swapchain_extent.width);
  viewport.height = static_cast<float>(ctx.swapchain_extent.height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

  VkRect2D scissor{};
  scissor.extent = ctx.swapchain_extent;
  vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

  // Набор сцены привязывается один раз на весь кадр
  vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelineLayout, 0, 1, &sceneSet, 0, nullptr);

  const VkDeviceSize offsets[] = {0};
  bool have_kind = false;
  DrawKind current_kind = DrawKind::Lit;

  for (const DrawCall &dc : draw_calls) {
    if (!have_kind || dc.kind != current_kind) {
      have_kind = true;
      current_kind = dc.kind;

      switch (dc.kind) {
      case DrawKind::Lit:
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          litPipeline);
        break;
      case DrawKind::UnlitSolid:
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          unlitSolidPipeline);
        break;
      case DrawKind::UnlitLines:
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          unlitLinePipeline);
        break;
      }

      if (dc.kind == DrawKind::UnlitLines) {
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1,
                               &coneVertexBuffer.buffer, offsets);
      } else {
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1,
                               &cubeVertexBuffer.buffer, offsets);
        vkCmdBindIndexBuffer(fd.command_buffer, cubeIndexBuffer.buffer, 0,
                             VK_INDEX_TYPE_UINT16);
      }
    }

    // Набор модели: один дескриптор, у каждого объекта своё смещение
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout, 1, 1, &modelSet, 1,
                            &dc.dynamicOffset);

    if (dc.kind == DrawKind::UnlitLines) {
      vkCmdDraw(fd.command_buffer, coneVertexCount, 1, 0, 0);
    } else {
      vkCmdDrawIndexed(fd.command_buffer, cubeIndexCount, 1, 0, 0, 0);
    }
  }

  vkCmdEndRenderPass(fd.command_buffer);
  vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application