#pragma once

#include "Core/Assert.hpp"
#include "Memory/Device.hpp"

#include "Ops/Backend/CPU/Matmul.hpp"
#include "Ops/Backend/Dispatch.hpp"

#include "Ops/Utils.hpp"
#include "Ops/Validation.hpp"

namespace ml::ops {
namespace detail {
inline core::Shape broadcast_shapes(const core::Shape &a, const core::Shape &b) {
  const auto rank = std::max(a.rank(), b.rank());
  std::vector<std::size_t> dims(rank);

  for (std::size_t i = 0; i < rank; i++) {
    const auto offsetA = rank - a.rank(), offsetB = rank - b.rank();
    const auto dimA = i < offsetA ? 1 : a[i - offsetA], dimB = i < offsetB ? 1 : b[i - offsetB];

    if (dimA != dimB && dimA != 1 && dimB != 1)
      throw ShapeError("Shapes are not broadcastle at dim {}: {} vs {}", i, dimA, dimB);

    dims[i] = std::max(dimA, dimB);
  }

  return core::Shape(std::move(dims));
}

inline core::Shape batch_shape(const core::Shape &shape) {
  if (shape.rank() < 2)
    throw ShapeError("batch_shape: expects at least rank 2 shape and got {}", shape.rank());

  std::vector<std::size_t> dims(shape.rank() - 2);
  for (std::size_t i = 0; i + 2 < shape.rank(); i++)
    dims[i] = shape[i];

  return core::Shape(std::move(dims));
}

inline std::size_t batch_offset(std::size_t idx, const core::Shape &batch_shape, const core::Strides &strides) {
  std::size_t offset = 0;
  for (std::size_t dim = batch_shape.rank(); dim-- > 0;) {
    const std::size_t i = idx % batch_shape[dim];
    idx /= batch_shape[dim];
    offset += i * strides[dim];
  }

  return offset;
}

inline core::Shape matmul_output_shape(const core::Shape &a, Transpose transA, const core::Shape &b, Transpose transB) {

  if (a.rank() < 2 || b.rank() < 2)
    throw ShapeError("matmul: expected rank >= 2, got {} and {}", a.rank(), b.rank());

  const std::size_t a0 = a[a.rank() - 2];
  const std::size_t a1 = a[a.rank() - 1];

  const std::size_t b0 = b[b.rank() - 2];
  const std::size_t b1 = b[b.rank() - 1];

  std::size_t N, K, K2, M;
  if (transA == Transpose::Yes)
    N = a1, K = a0;
  else
    N = a0, K = a1;

  if (transB == Transpose::Yes)
    K2 = b1, M = b0;
  else
    K2 = b0, M = b1;

  if (K != K2)
    throw ShapeError("matmul: incompatible dimensions {} and {}", K, K2);

  auto batches = broadcast_shapes(batch_shape(a), batch_shape(b));

  auto dims = batches.dims();
  dims.push_back(N);
  dims.push_back(M);

  return core::Shape(std::move(dims));
}
} // namespace detail

inline void matmul(const Tensor &A, Transpose transA, const Tensor &B, Transpose transB, bool overwrite, Tensor &out) {
  require_same_device(A, B, out);
  require_same_dtype(A, B, out);

  if (A.rank() < 2 || B.rank() < 2)
    throw ShapeError("batched_matmul: inputs must have rank >= 2; got {} and {}", A.rank(), B.rank());

  const auto &shapeA = A.shape(), &shapeB = B.shape();
  const auto a0 = shapeA[A.rank() - 2], a1 = shapeA[A.rank() - 1], b0 = shapeB[B.rank() - 2], b1 = shapeB[B.rank() - 1];

  std::size_t N, K, K2, M;
  if (transA == Transpose::Yes)
    N = a1, K = a0;
  else
    N = a0, K = a1;

  if (transB == Transpose::Yes)
    K2 = b1, M = b0;
  else
    K2 = b0, M = b1;

  if (K != K2)
    throw ShapeError("batched_matmul: incompatible matrix dimensions; A - [..., {}, {}], B - [..., {}, {}]", N, K, K2,
                     M);

  const auto batches = detail::broadcast_shapes(detail::batch_shape(shapeA), detail::batch_shape(shapeB));

  auto dimsA = batches.dims();
  dimsA.push_back(a0);
  dimsA.push_back(a1);

  auto dimsB = batches.dims();
  dimsB.push_back(b0);
  dimsB.push_back(b1);

  auto outDims = batches.dims();
  outDims.push_back(N);
  outDims.push_back(M);

  const core::Shape broadcastShapeA(std::move(dimsA)), broadcastShapeB(std::move(dimsB)),
      expectedOutShape(std::move(outDims));

  if (out.shape() != expectedOutShape)
    throw ShapeError("batched_matmul: output has incompatible shape; expected {}, got {}", expectedOutShape,
                     out.shape());

  const Tensor viewA = broadcast_to(A, broadcastShapeA), viewB = broadcast_to(B, broadcastShapeB);
  const std::size_t batch_rank = batches.rank();

  // viewA = [batch..., a0, a1]
  // viewB = [batch..., b0, b1]
  // op(A) = [N, K]
  // op(B) = [K, M]
  // out   = [batch..., N, M]

  const core::Strides matrixStridesA({viewA.strides()[batch_rank], viewA.strides()[batch_rank + 1]}),
      matrixStridesB({viewB.strides()[batch_rank], viewB.strides()[batch_rank + 1]}),
      matrixStridesOut({out.strides()[batch_rank], out.strides()[batch_rank + 1]});

  const core::Shape matrixShapeA{a0, a1}, matrixShapeB{b0, b1}, matrixShapeOut{N, M};

  const std::size_t batch_count = batch_rank == 0 ? 1 : batches.num_elements();
  dispatch(A.device(), A.dtype(), [&]<memory::DeviceType D, class T>() {
    if constexpr (D == memory::DeviceType::CPU) {
      for (std::size_t batch = 0; batch < batch_count; batch++) {
        const std::size_t offsetA = detail::batch_offset(batch, batches, viewA.strides()),
                          offsetB = detail::batch_offset(batch, batches, viewB.strides()),
                          offsetOut = detail::batch_offset(batch, batches, out.strides());

        Tensor matrixA = viewA.as_strided(matrixShapeA, matrixStridesA, offsetA),
               matrixB = viewB.as_strided(matrixShapeB, matrixStridesB, offsetB),
               matrixOut = out.as_strided(matrixShapeOut, matrixStridesOut, offsetOut);

        backend::cpu::matmul<T>(matrixA, transA, matrixB, transB, T{1}, overwrite ? T{0} : T{1}, matrixOut);
      }
    } else
      UNREACHABLE();
  });
}

[[nodiscard]] inline Tensor matmul(const Tensor &A, Transpose transA, const Tensor &B, Transpose transB) {
  require_same_device(A, B);
  require_same_dtype(A, B);

  Tensor out(detail::matmul_output_shape(A.shape(), transA, B.shape(), transB), A.dtype(), A.device());

  matmul(A, transA, B, transB, true, out);
  return out;
}
} // namespace ml::ops
