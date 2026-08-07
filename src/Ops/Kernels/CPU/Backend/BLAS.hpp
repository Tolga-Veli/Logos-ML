#pragma once

#include <cblas.h>

namespace ml::kernels::detail {
template <class T> struct BlasDispatch;

template <> struct BlasDispatch<float> {
  // Level 1
  static constexpr auto dot = cblas_sdot;
  static constexpr auto axpy = cblas_saxpy;
  static constexpr auto scal = cblas_sscal;
  static constexpr auto nrm2 = cblas_snrm2;
  static constexpr auto asum = cblas_sasum;
  static constexpr auto copy = cblas_scopy;
  // Level 2
  static constexpr auto gemv = cblas_sgemv;
  // Level 3
  static constexpr auto gemm = cblas_sgemm;

  // mixed precision: float inputs, double accumulation, double output
  static constexpr auto dsdot = cblas_dsdot;
};

template <> struct BlasDispatch<double> {
  // Level 1
  static constexpr auto dot = cblas_ddot;
  static constexpr auto axpy = cblas_daxpy;
  static constexpr auto scal = cblas_dscal;
  static constexpr auto nrm2 = cblas_dnrm2;
  static constexpr auto asum = cblas_dasum;
  static constexpr auto copy = cblas_dcopy;
  // Level 2
  static constexpr auto gemv = cblas_dgemv;
  // Level 3
  static constexpr auto gemm = cblas_dgemm;
};

} // namespace ml::kernels::detail
