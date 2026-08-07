#pragma once

#include "Core/DType.hpp"
#include "Memory/Device.hpp"
#include "Ops/Kernels/CPU/LinearAlgebra.hpp"
#include "Ops/Kernels/Dispatch.hpp"

namespace ml::ops {
using kernels::MatrixView;
using kernels::ScalarView;
using kernels::TensorView;
using kernels::VectorView;

inline void add_rowwise_vector(Tensor &mat, const Tensor &vec) {
  CORE_VERIFY(mat.rank() == 2 && vec.rank() == 1, "Invalid tensor arguments");
  CORE_VERIFY(mat.device() == vec.device() && mat.dtype() == vec.dtype(),
              "Invalid device/dtype");

  kernels::dispatch(mat.device(), mat.dtype(),
                    [&]<memory::DeviceType D, class T>() {
                      if constexpr (D == memory::DeviceType::CPU)
                        kernels::cpu::add_rowwise_vector_kernel<T>(
                            MatrixView<T>(mat), VectorView<const T>(vec));
                      else
                        CORE_VERIFY(false, "not implemented yet");
                    });
}

inline void sum_rows(const Tensor &mat, Tensor &vec) {
  CORE_VERIFY(mat.rank() == 2 && vec.rank() == 1, "Invalid tensor arguments");
  CORE_VERIFY(mat.device() == vec.device() && mat.dtype() == vec.dtype(),
              "Invalid device/dtype");

  kernels::dispatch(mat.device(), mat.dtype(),
                    [&]<memory::DeviceType D, class T>() {
                      if constexpr (D == memory::DeviceType::CPU)
                        kernels::cpu::sum_rows_kernel<T>(
                            MatrixView<const T>(mat), VectorView<T>(vec));
                      else
                        CORE_VERIFY(false, "not implemented yet");
                    });
}

// y := x   (plain copy, BLAS-optimized -- std::copy_n is an equally valid
// choice if x/y are both contiguous;
inline void copy(const Tensor &x, Tensor &y) {
  CORE_VERIFY(x.rank() == 1 && y.rank() == 1, "Invalid tensor arguments");
  CORE_VERIFY(x.device() == y.device(), "Invalid device");
  CORE_VERIFY(x.dtype() == y.dtype(), "Invalid dtype");

  kernels::dispatch(
      x.device(), x.dtype(), [&]<memory::DeviceType D, class T>() {
        if constexpr (D == memory::DeviceType::CPU)
          kernels::cpu::copy<T>(VectorView<const T>(x), VectorView<T>(y));
        else
          CORE_VERIFY(false, "not implemented yet");
      });
}

// the inner product <x,y> is in out
inline void dot(const Tensor &x, const Tensor &y, Tensor &z) {
  CORE_VERIFY(x.rank() == 1 && y.rank() == 1 && z.rank() == 0,
              "Invalid tensor arguments");
  CORE_VERIFY(x.device() == y.device() && x.device() == z.device(),
              "Invalid device");
  CORE_VERIFY(x.dtype() == y.dtype() && x.dtype() == z.dtype(),
              "Invalid dtype");

  kernels::dispatch(
      x.device(), x.dtype(), [&]<memory::DeviceType D, class T>() {
        if constexpr (D == memory::DeviceType::CPU)
          kernels::cpu::dot<T>(VectorView<const T>(x), VectorView<const T>(y),
                               ScalarView<T>(z));
        else
          CORE_VERIFY(false, "not implemented yet");
      });
}

// Returns the Euclidean (L2) norm of x, i.e. sqrt(sum(x_i^2))
inline void nrm2(const Tensor &x, Tensor &y) {
  CORE_VERIFY(x.rank() == 1 && y.rank() == 0, "Invalid tensor arguments");
  CORE_VERIFY(x.device() == y.device() && x.dtype() == y.dtype(),
              "Invalid device/dtype");

  kernels::dispatch(
      x.device(), x.dtype(), [&]<memory::DeviceType D, class T>() {
        if constexpr (D == memory::DeviceType::CPU)
          kernels::cpu::nrm2<T>(VectorView<const T>(x), ScalarView<T>(y));
        else
          CORE_VERIFY(false, "not implemented yet");
      });
}

// Returns the sum of absolute values, sum(|x_i|)  (the L1 norm)
inline void asum(const Tensor &x, Tensor &y) {
  CORE_VERIFY(x.rank() == 1 && y.rank() == 1, "Invalid tensor arguments");
  CORE_VERIFY(x.device() == y.device() && x.dtype() == y.dtype(),
              "Invalid device/dtype");

  kernels::dispatch(
      x.device(), x.dtype(), [&]<memory::DeviceType D, class T>() {
        if constexpr (D == memory::DeviceType::CPU)
          kernels::cpu::asum<T>(VectorView<const T>(x), ScalarView<T>(y));
        else
          CORE_VERIFY(false, "not implemented yet");
      });
}

// Returns x . y accumulated internally in double precision, even though
// x and y are float vectors. Meaningfully more accurate than dot<float>
// for long vectors, since plain float accumulation loses precision as
// the running sum grows relative to each incoming term.
//
// Only valid for T=float -- there's no extra precision tier above double
// to accumulate into, so this is a compile error for T=double rather
// than silently degrading to plain dot().

inline void dot_precise(const Tensor &x, const Tensor &y, Tensor &z) {
  CORE_VERIFY(x.rank() == 1 && y.rank() == 1 && z.rank() == 0,
              "Invalid tensor arguments");
  CORE_VERIFY(x.device() == y.device() && x.device() == z.device(),
              "Invalid device");
  CORE_VERIFY(x.dtype() == core::DType::Float32 &&
                  y.dtype() == core::DType::Float32 &&
                  z.dtype() == core::DType::Float64,
              "Invalid dtype");

  kernels::dispatch(x.device(), x.dtype(),
                    [&]<memory::DeviceType D, class T>() {
                      if constexpr (D == memory::DeviceType::CPU)
                        kernels::cpu::dot_precise<T>(VectorView<const T>(x),
                                                     VectorView<const T>(y),
                                                     ScalarView<double>(z));
                      else
                        return 0;
                    });
}
} // namespace ml::ops
