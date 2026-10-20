#pragma once

#include <cstdint>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

struct GLFWwindow;

namespace graphics::internal {

struct Context {
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue graphics_queue = VK_NULL_HANDLE;
  uint32_t graphics_queue_index = 0;
  VmaAllocator allocator = nullptr;
  VkFormat swapchain_format = VK_FORMAT_UNDEFINED;
  VkExtent2D swapchain_extent{};
  VkRenderPass render_pass = VK_NULL_HANDLE;
};

struct FrameData {
  VkFramebuffer framebuffer = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
};

extern Context context;

bool initialize(GLFWwindow *window);
void shutdown();
void resize(uint32_t width, uint32_t height);
FrameData prepare();
void submitAndPresent();

} // namespace graphics::internal
