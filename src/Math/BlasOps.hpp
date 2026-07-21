#pragma once

#include <cblas.h>

namespace ml::linalg {
enum class Transpose { No = 0, Yes };
enum class Triangular { Upper = 0, Lower };
enum class Side { Left = 0, Right };
enum class Diagonal { NonUnit = 0, Unit };

namespace detail {
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
  static constexpr auto symv = cblas_ssymv;
  static constexpr auto trmv = cblas_strmv;
  static constexpr auto trsv = cblas_strsv;
  static constexpr auto ger = cblas_sger;
  // Level 3
  static constexpr auto gemm = cblas_sgemm;
  static constexpr auto symm = cblas_ssymm;
  static constexpr auto syrk = cblas_ssyrk;
  static constexpr auto syr2k = cblas_ssyr2k;
  static constexpr auto trmm = cblas_strmm;
  static constexpr auto trsm = cblas_strsm;
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
  static constexpr auto symv = cblas_dsymv;
  static constexpr auto trmv = cblas_dtrmv;
  static constexpr auto trsv = cblas_dtrsv;
  static constexpr auto ger = cblas_dger;
  // Level 3
  static constexpr auto gemm = cblas_dgemm;
  static constexpr auto symm = cblas_dsymm;
  static constexpr auto syrk = cblas_dsyrk;
  static constexpr auto syr2k = cblas_dsyr2k;
  static constexpr auto trmm = cblas_dtrmm;
  static constexpr auto trsm = cblas_dtrsm;
};

inline constexpr auto dsdot = cblas_dsdot;

} // namespace detail
} // namespace ml::linalg
