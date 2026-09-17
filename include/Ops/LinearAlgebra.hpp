#pragma once

#include "Core/DType.hpp"
#include "Memory/Device.hpp"

#include "Ops/Backend/CPU/LinearAlgebra.hpp"
#include "Ops/Backend/Dispatch.hpp"

#include "Ops/Validation.hpp"

namespace ml::ops {
inline void add_rowwise_vector(core::Tensor &mat, const core::Tensor &vec) {
  require_same_device(mat, vec);
  require_same_dtype(mat, vec);

  dispatch(mat.device(), mat.dtype(), [&]<memory::DeviceType D, class T>() {
    if constexpr (D == memory::DeviceType::CPU)
      backend::cpu::add_rowwise_vector<T>(mat, vec);
    else
      UNREACHABLE();
  });
}

inline void sum_rows(const core::Tensor &mat, core::Tensor &vec) {
  require_same_device(mat, vec);
  require_same_dtype(mat, vec);

  dispatch(mat.device(), mat.dtype(), [&]<memory::DeviceType D, class T>() {
    if constexpr (D == memory::DeviceType::CPU)
      backend::cpu::sum_rows<T>(mat, vec);
    else
      UNREACHABLE();
  });
}

// y := x (BLAS-optimized and supports strided vectors).
inline void copy(const core::Tensor &x, core::Tensor &y) {
  require_same_device(x, y);
  require_same_dtype(x, y);

  dispatch(x.device(), x.dtype(), [&]<memory::DeviceType D, class T>() {
    if constexpr (D == memory::DeviceType::CPU)
      backend::cpu::copy<T>(x, y);
    else
      UNREACHABLE();
  });
}

// Stores the inner product <x, y> in z.
inline void dot(const core::Tensor &x, const core::Tensor &y, core::Tensor &z) {
  require_same_device(x, y, z);
  require_same_dtype(x, y, z);

  dispatch(x.device(), x.dtype(), [&]<memory::DeviceType D, class T>() {
    if constexpr (D == memory::DeviceType::CPU)
      backend::cpu::dot<T>(x, y, z);
    else
      UNREACHABLE();
  });
}

// Stores the Euclidean (L2) norm sqrt(sum(x_i^2)) in y.
inline void nrm2(const core::Tensor &x, core::Tensor &y) {
  require_same_device(x, y);
  require_same_dtype(x, y);

  dispatch(x.device(), x.dtype(), [&]<memory::DeviceType D, class T>() {
    if constexpr (D == memory::DeviceType::CPU)
      backend::cpu::nrm2<T>(x, y);
    else
      UNREACHABLE();
  });
}

// Stores the sum of absolute values, sum(|x_i|), in y.
inline void asum(const core::Tensor &x, core::Tensor &y) {
  require_same_device(x, y);
  require_same_dtype(x, y);

  dispatch(x.device(), x.dtype(), [&]<memory::DeviceType D, class T>() {
    if constexpr (D == memory::DeviceType::CPU)
      backend::cpu::asum<T>(x, y);
    else
      UNREACHABLE();
  });
}

// Returns dot(x,y) accumulated internally in double precision, even though
// x and y are float vectors. Meaningfully more accurate than dot<float>
// for long vectors, since plain float accumulation loses precision as
// the running sum grows relative to each incoming term.
//
// Only valid for T=float -- there's no extra precision tier above double
// to accumulate into, so this is a compile error for T=double rather
// than silently degrading to plain dot().

inline void dot_precise(const core::Tensor &x, const core::Tensor &y, core::Tensor &z) {
  require_same_device(x, y, z);
  if (x.dtype() != core::DType::Float32 || y.dtype() != core::DType::Float32 || z.dtype() != core::DType::Float64)
    throw DTypeError("dot_precise: expected Float32, Float32, Float64 and got: {} {} {}", x.dtype(), y.dtype(),
                     z.dtype());

  dispatch(x.device(), x.dtype(), [&]<memory::DeviceType D, class T>() {
    if constexpr (D == memory::DeviceType::CPU)
      backend::cpu::dot_precise<T>(x, y, z);
    else
      UNREACHABLE();
  });
}

inline void add_inplace(const core::Tensor &in, core::Tensor &out) {
  if (in.shape() != out.shape())
    throw ShapeError("add_inplace: Shape mismatch");

  require_same_device(in, out);
  require_same_dtype(in, out);

  dispatch(in.device(), out.dtype(), [&]<memory::DeviceType type, class T>() {
    if constexpr (type == memory::DeviceType::CPU)
      backend::cpu::add_inplace<T>(in, out);
    else
      UNREACHABLE();
  });
}

} // namespace ml::ops
