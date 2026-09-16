#pragma once

#include <cblas.h>

namespace ml::backend::detail {
template <class T> struct BlasDispatch;

template <> struct BlasDispatch<float> {
  static constexpr auto dot = cblas_sdot;
  static constexpr auto axpy = cblas_saxpy;
  static constexpr auto scal = cblas_sscal;
  static constexpr auto nrm2 = cblas_snrm2;
  static constexpr auto asum = cblas_sasum;
  static constexpr auto copy = cblas_scopy;

  static constexpr auto gemv = cblas_sgemv;
  static constexpr auto gemm = cblas_sgemm;

  // mixed precision: float inputs, double accumulation, double output
  static constexpr auto dsdot = cblas_dsdot;
};

template <> struct BlasDispatch<double> {
  static constexpr auto dot = cblas_ddot;
  static constexpr auto axpy = cblas_daxpy;
  static constexpr auto scal = cblas_dscal;
  static constexpr auto nrm2 = cblas_dnrm2;
  static constexpr auto asum = cblas_dasum;
  static constexpr auto copy = cblas_dcopy;

  static constexpr auto gemv = cblas_dgemv;

  static constexpr auto gemm = cblas_dgemm;
};

} // namespace ml::backend::detail
