#pragma once

#include <type_traits>

namespace ml::backend {

// NOTE: if adding custom types needs to be specified in the concept
template <class T>
concept ViewBaseType = std::is_floating_point_v<T> || std::is_integral_v<T>;

} // namespace ml::backend
