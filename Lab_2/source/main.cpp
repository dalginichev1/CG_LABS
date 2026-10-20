#include <climits>
#include <cstdint>

#include <array>
#include <exception>
#include <filesystem>
#include <iostream>
#include <system_error>
#include <vector>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <backends/imgui_impl_glfw.h>
#include <imgui.h>

#include "application.hpp"
#include "graphics_internal.hpp"

namespace {

namespace fs = std::filesystem;

bool hasRequiredShaders(const fs::path &root) {
  constexpr std::array<const char *, 4> shader_files = {
      "shader.vert.spv", "shader.frag.spv", "unlit.vert.spv", "unlit.frag.spv"};

  for (const char *file : shader_files) {
    std::error_code ec;
    if (!fs::is_regular_file(root / "shaders" / file, ec) || ec)
      return false;
  }
  return true;
}

bool locateShaderDirectory(const char *argv0) {
  std::vector<fs::path> roots;
  std::error_code ec;

  fs::path cwd = fs::current_path(ec);
  if (!ec)
    roots.push_back(cwd);

  if (argv0 != nullptr && *argv0 != '\0') {
    ec.clear();
    fs::path executable = fs::absolute(fs::path(argv0), ec);
    if (!ec)
      roots.push_back(executable.parent_path());
  }

  std::vector<fs::path> candidates;
  for (fs::path root : roots) {
    for (int depth = 0; depth < 8 && !root.empty(); ++depth) {
      candidates.push_back(root);
      fs::path parent = root.parent_path();
      if (parent == root)
        break;
      root = parent;
    }
  }

  for (const fs::path &candidate : candidates) {
    if (!hasRequiredShaders(candidate))
      continue;

    ec.clear();
    fs::current_path(candidate, ec);
    if (!ec) {
      std::cout << "Каталог шейдеров: " << candidate.string() << '\n';
      return true;
    }
  }

  std::cerr << "Не найдены скомпилированные SPIR-V-шейдеры. Ожидаются файлы "
               "shaders/shader.vert.spv, shader.frag.spv, unlit.vert.spv и "
               "unlit.frag.spv. Запустите compile_shaders.bat из каталога "
               "проекта, затем запустите приложение снова.\n";
  return false;
}

constexpr int32_t default_window_width = 1280;
constexpr int32_t default_window_height = 720;

constexpr char default_window_title[] = "Vulkan Starter App";

GLFWwindow *glfw_window;

} // namespace

int main(int argc, char **argv) {
  int status = EXIT_SUCCESS;

  if (!locateShaderDirectory(argc > 0 ? argv[0] : nullptr))
    return EXIT_FAILURE;

  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW\n";
    return EXIT_FAILURE;
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  glfw_window = glfwCreateWindow(default_window_width, default_window_height,
                                 default_window_title, nullptr, nullptr);
  if (glfw_window == nullptr) {
    status = EXIT_FAILURE;
    goto err_null_window;
  }

  glfwSetFramebufferSizeCallback(glfw_window,
                                 [](GLFWwindow *, int width, int height) {
                                   if (width == 0 || height == 0) {
                                     return;
                                   }

                                   graphics::internal::resize(width, height);
                                 });

  if (ImGui::CreateContext() == nullptr) {
    std::cerr << "Failed to create ImGUI context\n";
    status = EXIT_FAILURE;
    goto err_imgui_init;
  }

  {
    ImGuiIO &io = ImGui::GetIO();
#if defined(_WIN32)
    io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\arial.ttf", 16.0f,
                                 nullptr, io.Fonts->GetGlyphRangesCyrillic());
#endif
  }

  if (!ImGui_ImplGlfw_InitForVulkan(glfw_window, true)) {
    std::cerr
        << "Failed to initialize ImGUI GLFW backend for Vulkan renderer\n";
    status = EXIT_FAILURE;
    goto err_imgui_glfw_init;
  }

  if (!graphics::internal::initialize(glfw_window)) {
    std::cerr << "Failed to initialize graphics\n";
    status = EXIT_FAILURE;
    goto err_graphics_init;
  }

  try {
    if (!application::initialize()) {
      std::cerr << "Failed to initialize application\n";
      status = EXIT_FAILURE;
      goto err_application_init;
    }
  } catch (const std::exception &e) {
    std::cerr << "Ошибка инициализации приложения: " << e.what() << '\n';
    application::shutdown();
    status = EXIT_FAILURE;
    goto err_application_init;
  } catch (...) {
    std::cerr << "Неизвестная ошибка при инициализации приложения\n";
    application::shutdown();
    status = EXIT_FAILURE;
    goto err_application_init;
  }

  while (!glfwWindowShouldClose(glfw_window)) {
    const double time = glfwGetTime();

    glfwPollEvents();
    ImGui_ImplGlfw_NewFrame();

    ImGui::NewFrame();
    application::update(time);
    ImGui::Render();

    graphics::internal::FrameData fd = graphics::internal::prepare();
    application::render(fd);
    graphics::internal::submitAndPresent();
  }

  application::shutdown();
err_application_init:
  graphics::internal::shutdown();
err_graphics_init:
  ImGui_ImplGlfw_Shutdown();
err_imgui_glfw_init:
  ImGui::DestroyContext();
err_imgui_init:
  glfwDestroyWindow(glfw_window);
err_null_window:
  glfwTerminate();

  return status;
}
