#pragma once

#include <optional>
#include <utility>

#include "Core/Tensor.hpp"
#include "Ops/LinearAlgebra.hpp"

namespace ml::core {

class Parameter final {
public:
  explicit Parameter(Tensor _data, bool requires_grad = true)
      : m_Data(std::move(_data)), m_RequiresGrad(requires_grad) {}

  Parameter(const Parameter &) = delete;
  Parameter &operator=(const Parameter &) = delete;
  Parameter(Parameter &&) = delete;
  Parameter &operator=(Parameter &&) = delete;

  [[nodiscard]] Tensor &value() noexcept { return m_Data; }
  [[nodiscard]] const Tensor &value() const noexcept { return m_Data; }
  [[nodiscard]] bool requires_grad() const noexcept { return m_RequiresGrad; }

  void initialize_grad(Tensor tensor) { m_Grad = tensor; }

  void set_requires_grad(bool fl) noexcept {
    m_RequiresGrad = fl;
    if (!fl)
      clear_grad();
  }

  [[nodiscard]] bool has_grad() const noexcept { return m_Grad.has_value(); }

  [[nodiscard]] Tensor &grad() {
    if (!m_Grad.has_value())
      throw std::logic_error("Parameter: empty grad tensor");

    return m_Grad.value();
  }

  [[nodiscard]] const Tensor &grad() const {
    if (!m_Grad.has_value())
      throw std::logic_error("Parameter: empty grad tensor");

    return m_Grad.value();
  }

  void accumulate_grad(const Tensor &tensor) {
    if (!m_RequiresGrad)
      return;

    if (tensor.shape() != m_Data.shape())
      throw ShapeError("Parameter: gradient shape mismatch");

    if (has_grad())
      ops::add_inplace(tensor, m_Grad.value());
    else
      m_Grad.emplace(tensor.clone());
  }

  void zero_grad() {
    if (m_Grad.has_value())
      ops::fill_zeroes(m_Grad.value());
  }

  void clear_grad() noexcept { m_Grad.reset(); }

private:
  Tensor m_Data;
  std::optional<Tensor> m_Grad;
  bool m_RequiresGrad;
};

} // namespace ml::core
