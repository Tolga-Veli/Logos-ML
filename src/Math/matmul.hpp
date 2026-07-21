#include "Core/MatrixView.hpp"

namespace ml::linalg {
template <class T> using MatrixView = core::MatrixView<T>;

namespace detail {

// C = alpha * A * B + beta * C
// A: M x K
// B: K x N
// C: M x N
template <class T>
void gemm_generic(T alpha, MatrixView<const T> &A, MatrixView<const T> &B,
                  T beta, MatrixView<T> out) {

  const std::size_t M = A.rows(), N = B.cols(), K = A.cols();
  for (std::size_t i{0}; i < M; i++)
    for (std::size_t j{0}; j < N; j++) {
      T sum{0};

      for (std::size_t k{0}; k < K; k++)
        sum += A(i, k) * B(k, j);

      out(i, j) = alpha * sum + beta * out(i, j);
    }
}

template <class T>
void gemm_row_major(T alpha, MatrixView<const T> &A, MatrixView<const T> &B,
                    T beta, MatrixView<T> out) {

  const std::size_t M = A.rows(), N = B.cols(), K = A.cols();
  for (std::size_t i{0}; i < M; i++) {
    for (std::size_t j{0}; j < N; j++)
      out(i, j) *= beta;

    for (std::size_t k{0}; k < K; k++) {
      const T a = alpha * A(i, k);
      for (std::size_t j{0}; j < N; j++)
        out(i, j) += a * B(k, j);
    }
  }
}

} // namespace detail

template <class T>
void gemm(T alpha, MatrixView<const T> &A, MatrixView<const T> &B, T beta,
          MatrixView<T> out) {
  assert(A.cols() == B.rows());
  assert(out.rows() == A.rows());
  assert(out.cols() == B.cols());
}

} // namespace ml::linalg
