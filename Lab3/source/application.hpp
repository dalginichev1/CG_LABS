#pragma once

#include "graphics_internal.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace application {

bool initialize();
void update(double time);
void render(const graphics::internal::FrameData &frame_data);
void shutdown();

} // namespace application
