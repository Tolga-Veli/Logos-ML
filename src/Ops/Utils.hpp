#pragma once

#include "Core/Tensor.hpp"

#include <cblas.h>

namespace ml::ops {
using core::Tensor;

enum class Transpose { No = 0, Yes };
enum class Triangular { Upper = 0, Lower };
enum class Side { Left = 0, Right };
enum class Diagonal { NonUnit = 0, Unit };

} // namespace ml::ops
