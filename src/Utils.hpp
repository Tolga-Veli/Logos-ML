#pragma once

#include <string>
#include <filesystem>
#include <fstream>
#include <cassert>

namespace logos::utils {

inline std::string ReadFile(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  assert(in && "Failed to open file.");

  in.seekg(0, std::ios::end);
  std::string contents;
  contents.resize(static_cast<std::size_t>(in.tellg()));
  in.seekg(0, std::ios::beg);
  in.read(contents.data(), contents.size());

  assert(in && "Failed to read file.");
  return contents;
}
} // namespace logos::utils