#pragma once

#include "graphics_internal.hpp"
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace application {

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

bool initialize();
void shutdown();

void update(double time);
void render(const graphics::internal::FrameData &fd);

} // namespace application