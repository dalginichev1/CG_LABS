#include "application.hpp"
#include <imgui.h>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace application {
namespace {

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
};

struct UBO {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
  glm::vec4 baseColor;
};

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 0.0f}},
    {{-0.5f, 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{0.5f, -0.5f, 0.5f}, {1.0f, 0.0f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 1.0f}}};

const std::vector<uint16_t> indices = {0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4,
                                       0, 4, 7, 7, 3, 0, 1, 5, 6, 6, 2, 1,
                                       3, 2, 6, 6, 7, 3, 0, 1, 5, 5, 4, 0};

VkPipelineLayout pipelineLayout;
VkPipeline graphicsPipeline;
VkDescriptorSetLayout descriptorSetLayout;
VkDescriptorPool descriptorPool;
constexpr size_t kObjectCount = 3;
VkDescriptorSet descriptorSets[kObjectCount];
VkBuffer vertexBuffer;
VmaAllocation vertexBufferAllocation;
VkBuffer indexBuffer;
VmaAllocation indexBufferAllocation;
VkBuffer uniformBuffers[kObjectCount];
VmaAllocation uniformBuffersAllocation[kObjectCount];
void *uniformBuffersMapped[kObjectCount];

bool use_perspective = true;
double last_time = 0.0;

struct CubeState {
  glm::vec3 pos;
  glm::vec3 rot;
  glm::vec3 scale = {1.0f, 1.0f, 1.0f};
  float color[3] = {1.0f, 1.0f, 1.0f};

  bool animate = false;
  float anim_speed = 1.0f;
  float anim_radius = 2.0f;
  float anim_time = 0.0f;
};

CubeState cube1;
CubeState cube2{.pos = {2.0f, 0.0f, 0.0f}};

void drawCubeUI(const char *title, const char *id_suffix, CubeState &cube) {
  ImGui::PushID(id_suffix);
  ImGui::Text("%s", title);

  ImGui::Text("Трансформации (Доп. 2)");
  ImGui::DragFloat3("Позиция", glm::value_ptr(cube.pos), 0.05f);
  ImGui::DragFloat3("Поворот", glm::value_ptr(cube.rot), 0.05f);
  ImGui::DragFloat3("Масштаб", glm::value_ptr(cube.scale), 0.05f);

  ImGui::Text("Анимация (Доп. 3)");
  ImGui::Checkbox("Воспроизведение (по кругу)", &cube.animate);
  ImGui::SliderFloat("Скорость", &cube.anim_speed, 0.1f, 5.0f);
  ImGui::SliderFloat("Радиус траектории", &cube.anim_radius, 0.0f, 10.0f);

  ImGui::Text("Цвет фигуры (Доп. 4 и 5)");
  ImGui::ColorEdit3("Базовый цвет", cube.color);

  ImGui::PopID();
}

void updateCubeAnimation(CubeState &cube, float delta_time) {
  if (!cube.animate)
    return;
  cube.anim_time += delta_time * cube.anim_speed;
  cube.pos.x = std::cos(cube.anim_time) * cube.anim_radius;
  cube.pos.z = std::sin(cube.anim_time) * cube.anim_radius;
  cube.rot.y += delta_time * cube.anim_speed;
  cube.rot.x += delta_time * (cube.anim_speed * 0.5f);
}

glm::mat4 cubeModelMatrix(const CubeState &cube) {
  glm::mat4 model = glm::translate(glm::mat4(1.0f), cube.pos);
  model = glm::rotate(model, cube.rot.x, glm::vec3(1.0f, 0.0f, 0.0f));
  model = glm::rotate(model, cube.rot.y, glm::vec3(0.0f, 1.0f, 0.0f));
  model = glm::rotate(model, cube.rot.z, glm::vec3(0.0f, 0.0f, 1.0f));
  model = glm::scale(model, cube.scale);
  return model;
}

std::vector<char> readFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("Не удалось открыть файл шейдера: " + filename);
  size_t fileSize = static_cast<size_t>(file.tellg());
  std::vector<char> buffer(fileSize);
  file.seekg(0);
  file.read(buffer.data(), fileSize);
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
  VkResult result =
      vkCreateShaderModule(ctx.device, &createInfo, nullptr, &shaderModule);
  if (result != VK_SUCCESS)
    throw std::runtime_error("Failed to create shader module!");
  return shaderModule;
}

void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer &buffer,
                  VmaAllocation &allocation,
                  VmaAllocationInfo *allocInfoOut = nullptr) {
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

  VkResult result = vmaCreateBuffer(ctx.allocator, &bufferInfo, &vmaAllocInfo,
                                    &buffer, &allocation, allocInfoOut);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan buffer! VkResult = " +
                             std::to_string(result));
  }
}

}

bool initialize() {
  auto &ctx = graphics::internal::context;

  VmaAllocationInfo vboAllocInfo, iboAllocInfo;
  createBuffer(sizeof(vertices[0]) * vertices.size(),
               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer,
               vertexBufferAllocation, &vboAllocInfo);
  memcpy(vboAllocInfo.pMappedData, vertices.data(),
         sizeof(vertices[0]) * vertices.size());

  createBuffer(sizeof(indices[0]) * indices.size(),
               VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer,
               indexBufferAllocation, &iboAllocInfo);
  memcpy(iboAllocInfo.pMappedData, indices.data(),
         sizeof(indices[0]) * indices.size());

  for (size_t i = 0; i < kObjectCount; i++) {
    VmaAllocationInfo uboAllocInfo;
    createBuffer(sizeof(UBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 uniformBuffers[i], uniformBuffersAllocation[i], &uboAllocInfo);
    uniformBuffersMapped[i] = uboAllocInfo.pMappedData;
  }

  VkDescriptorSetLayoutBinding uboLayoutBinding{};
  uboLayoutBinding.binding = 0;
  uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  uboLayoutBinding.descriptorCount = 1;

  uboLayoutBinding.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo layoutInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layoutInfo.bindingCount = 1;
  layoutInfo.pBindings = &uboLayoutBinding;
  if (vkCreateDescriptorSetLayout(ctx.device, &layoutInfo, nullptr,
                                  &descriptorSetLayout) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create descriptor set layout!");
  }

  VkDescriptorPoolSize poolSize{};
  poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSize.descriptorCount = kObjectCount;

  VkDescriptorPoolCreateInfo poolInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  poolInfo.maxSets = kObjectCount;
  if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr,
                             &descriptorPool) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create descriptor pool!");
  }

  std::vector<VkDescriptorSetLayout> layouts(kObjectCount, descriptorSetLayout);
  VkDescriptorSetAllocateInfo allocInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocInfo.descriptorPool = descriptorPool;
  allocInfo.descriptorSetCount = kObjectCount;
  allocInfo.pSetLayouts = layouts.data();
  if (vkAllocateDescriptorSets(ctx.device, &allocInfo, descriptorSets) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate descriptor sets!");
  }

  for (size_t i = 0; i < kObjectCount; i++) {
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = uniformBuffers[i];
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(UBO);

    VkWriteDescriptorSet descriptorWrite{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    descriptorWrite.dstSet = descriptorSets[i];
    descriptorWrite.dstBinding = 0;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pBufferInfo = &bufferInfo;
    vkUpdateDescriptorSets(ctx.device, 1, &descriptorWrite, 0, nullptr);
  }

  auto vertShaderCode = readFile("shaders/shader.vert.spv");
  auto fragShaderCode = readFile("shaders/shader.frag.spv");
  VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
  VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

  VkPipelineShaderStageCreateInfo vertShaderStageInfo{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertShaderStageInfo.module = vertShaderModule;
  vertShaderStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo fragShaderStageInfo{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragShaderStageInfo.module = fragShaderModule;
  fragShaderStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                    fragShaderStageInfo};

  VkVertexInputBindingDescription bindingDescription{};
  bindingDescription.binding = 0;
  bindingDescription.stride = sizeof(Vertex);
  bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  VkVertexInputAttributeDescription attributeDescriptions[2]{};
  attributeDescriptions[0].binding = 0;
  attributeDescriptions[0].location = 0;
  attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[0].offset = offsetof(Vertex, pos);
  attributeDescriptions[1].binding = 0;
  attributeDescriptions[1].location = 1;
  attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[1].offset = offsetof(Vertex, color);

  VkPipelineVertexInputStateCreateInfo vertexInputInfo{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vertexInputInfo.vertexBindingDescriptionCount = 1;
  vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
  vertexInputInfo.vertexAttributeDescriptionCount = 2;
  vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
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

  std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                               VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamicState{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
  dynamicState.pDynamicStates = dynamicStates.data();

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
  vkCreatePipelineLayout(ctx.device, &pipelineLayoutInfo, nullptr,
                         &pipelineLayout);

  VkGraphicsPipelineCreateInfo pipelineInfo{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = shaderStages;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
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

  VkResult result = vkCreateGraphicsPipelines(
      ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline);
  if (result != VK_SUCCESS) {
    std::cerr << "Failed to create graphics pipeline! VkResult = " << result
              << std::endl;
  }

  vkDestroyShaderModule(ctx.device, fragShaderModule, nullptr);
  vkDestroyShaderModule(ctx.device, vertShaderModule, nullptr);

  return true;
}

void shutdown() {
  auto &ctx = graphics::internal::context;
  vkDeviceWaitIdle(ctx.device);
  vkDestroyPipeline(ctx.device, graphicsPipeline, nullptr);
  vkDestroyPipelineLayout(ctx.device, pipelineLayout, nullptr);
  vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(ctx.device, descriptorSetLayout, nullptr);
  vmaDestroyBuffer(ctx.allocator, vertexBuffer, vertexBufferAllocation);
  vmaDestroyBuffer(ctx.allocator, indexBuffer, indexBufferAllocation);
  for (size_t i = 0; i < kObjectCount; i++) {
    vmaDestroyBuffer(ctx.allocator, uniformBuffers[i],
                     uniformBuffersAllocation[i]);
  }
}

void update(double time) {
  if (last_time == 0.0)
    last_time = time;
  float delta_time = static_cast<float>(time - last_time);
  last_time = time;

  ImGui::Begin("Настройки 3D Сцены");

  ImGui::Text("Проекция (Доп. 1)");
  if (ImGui::RadioButton("Perspective Projection", use_perspective))
    use_perspective = true;
  ImGui::SameLine();
  if (ImGui::RadioButton("Orthographic Projection", !use_perspective))
    use_perspective = false;
  ImGui::Separator();

  drawCubeUI("Первый куб", "cube1", cube1);
  ImGui::Separator();
  drawCubeUI("Второй куб", "cube2", cube2);

  ImGui::End();

  updateCubeAnimation(cube1, delta_time);
  updateCubeAnimation(cube2, delta_time);

  auto &ctx = graphics::internal::context;
  float aspect = static_cast<float>(ctx.swapchain_extent.width) /
                 static_cast<float>(ctx.swapchain_extent.height);

  glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 2.5f, 4.5f),
                               glm::vec3(0.0f, 0.0f, 0.0f),
                               glm::vec3(0.0f, 1.0f, 0.0f)
  );

  glm::mat4 proj;
  if (use_perspective) {
    proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
  } else {
    float ortho_s = 4.0f;
    proj = glm::ortho(-aspect * ortho_s, aspect * ortho_s, -ortho_s, ortho_s,
                      0.1f, 100.0f);
  }
  proj[1][1] *= -1.0f;

  UBO ubo1{};
  ubo1.model = cubeModelMatrix(cube1);
  ubo1.view = view;
  ubo1.proj = proj;
  ubo1.baseColor = glm::vec4(cube1.color[0], cube1.color[1], cube1.color[2], 1.0f);
  memcpy(uniformBuffersMapped[0], &ubo1, sizeof(ubo1));

  glm::mat4 model2 =
      glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1.2f, 0.0f));
  model2 = glm::scale(model2, glm::vec3(5.0f, 0.3f, 5.0f));

  UBO ubo2{};
  ubo2.model = model2;
  ubo2.view = view;
  ubo2.proj = proj;
  ubo2.baseColor = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
  memcpy(uniformBuffersMapped[1], &ubo2, sizeof(ubo2));

  UBO ubo3{};
  ubo3.model = cubeModelMatrix(cube2);
  ubo3.view = view;
  ubo3.proj = proj;
  ubo3.baseColor = glm::vec4(cube2.color[0], cube2.color[1], cube2.color[2], 1.0f);
  memcpy(uniformBuffersMapped[2], &ubo3, sizeof(ubo3));
}

void render(const graphics::internal::FrameData &fd) {
  auto &ctx = graphics::internal::context;

  VkCommandBufferBeginInfo beginInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(fd.command_buffer, &beginInfo);

  VkClearValue clearValues[2] = {};
  clearValues[0].color = {{0.1f, 0.1f, 0.15f, 1.0f}};
  clearValues[1].depthStencil = {1.0f, 0};

  VkRenderPassBeginInfo renderPassInfo{
      VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  renderPassInfo.renderPass = ctx.render_pass;
  renderPassInfo.framebuffer = fd.framebuffer;
  renderPassInfo.renderArea.offset = {0, 0};
  renderPassInfo.renderArea.extent = ctx.swapchain_extent;
  renderPassInfo.clearValueCount = 2;
  renderPassInfo.pClearValues = clearValues;

  vkCmdBeginRenderPass(fd.command_buffer, &renderPassInfo,
                       VK_SUBPASS_CONTENTS_INLINE);

  vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    graphicsPipeline);

  VkViewport viewport{};
  viewport.width = static_cast<float>(ctx.swapchain_extent.width);
  viewport.height = static_cast<float>(ctx.swapchain_extent.height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

  VkRect2D scissor{};
  scissor.extent = ctx.swapchain_extent;
  vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertexBuffer, offsets);
  vkCmdBindIndexBuffer(fd.command_buffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

  vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelineLayout, 0, 1, &descriptorSets[0], 0, nullptr);
  vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(indices.size()), 1,
                   0, 0, 0);

  vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelineLayout, 0, 1, &descriptorSets[1], 0, nullptr);
  vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(indices.size()), 1,
                   0, 0, 0);

  vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelineLayout, 0, 1, &descriptorSets[2], 0, nullptr);
  vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(indices.size()), 1,
                   0, 0, 0);

  vkCmdEndRenderPass(fd.command_buffer);
  vkEndCommandBuffer(fd.command_buffer);
}

}
