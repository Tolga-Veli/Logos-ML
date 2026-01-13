#pragma once

#include <string>
#include <filesystem>
#include <fstream>
#include <cassert>

#include "Logging.hpp"

namespace logos::utils {

inline std::string ReadFile(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    L_ERROR("Failed to open file: " + path.string());
    return "";
  }

  in.seekg(0, std::ios::end);
  std::string contents;
  contents.resize(static_cast<std::size_t>(in.tellg()));
  in.seekg(0, std::ios::beg);
  in.read(contents.data(), contents.size());

  if (!in) {
    L_ERROR("Failed to read file: " + path.string());
    return "";
  }
  return contents;
}
} // namespace logos::utils