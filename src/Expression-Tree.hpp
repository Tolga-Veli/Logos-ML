#pragma once

#include <cassert>
#include <cstdint>
#include <memory>

namespace logos {
template <typename value_type = double> class ExpressionTree {
public:
  enum class NodeType : uint8_t {
    None = 0,
    Value,
    Addition,
    Subtraction,
    Multiplication,
    Division
  };

  struct Node {
    Node() : type(NodeType::None), data() {}
    explicit Node(value_type _data) : type(NodeType::Value), data(_data) {}
    explicit Node(NodeType _type, std::unique_ptr<Node> l,
                  std::unique_ptr<Node> r)
        : type(_type), left(std::move(l)), right(std::move(r)) {}

    NodeType type;
    value_type data;
    std::unique_ptr<Node> left{}, right{};

    static std::unique_ptr<Node> MakeValue(const value_type &v) {
      return std::make_unique<Node>(v);
    }

    static std::unique_ptr<Node>
    MakeBinary(NodeType op, std::unique_ptr<Node> l, std::unique_ptr<Node> r) {
      assert(op != NodeType::Value && op != NodeType::None &&
             "Invalid NodeType for MakeOp()");
      assert(l && r && "Invalid left or right pointer when creating Node");
      return std::make_unique<Node>(op, std::move(l), std::move(r));
    }

    value_type evaluate() {
      switch (type) {
      case NodeType::Value:
        return data;
      case NodeType::Addition:
        return left->evaluate() + right->evaluate();
      case NodeType::Subtraction:
        return left->evaluate() - right->evaluate();
      case NodeType::Multiplication:
        return left->evaluate() * right->evaluate();
      case NodeType::Division: {
        value_type L = left->evaluate(), R = right->evaluate();
        assert(R != value_type{} && "Trying to divide by zero!");
        return L / R;
      }
      case NodeType::None:
      default:
        return value_type{};
      }
    }
  };

  ExpressionTree() = default;
  ~ExpressionTree() = default;

  ExpressionTree(ExpressionTree &other) = delete;
  ExpressionTree &operator=(ExpressionTree &) = delete;

  ExpressionTree(ExpressionTree &&other) noexcept = default;
  ExpressionTree &operator=(ExpressionTree &&other) noexcept = default;

  value_type Evaluate() {
    assert(m_Root && "The root of the expression tree was a nullptr");
    return m_Root->evaluate();
  }

  inline static std::unique_ptr<Node> Value(const value_type &v) {
    return Node::MakeValue(v);
  }

  inline static std::unique_ptr<Node> Add(std::unique_ptr<Node> left,
                                          std::unique_ptr<Node> right) {
    return Node::MakeBinary(NodeType::Addition, std::move(left),
                            std::move(right));
  }

  inline static std::unique_ptr<Node> Sub(std::unique_ptr<Node> left,
                                          std::unique_ptr<Node> right) {
    return Node::MakeBinary(NodeType::Subtraction, std::move(left),
                            std::move(right));
  }

  inline static std::unique_ptr<Node> Mul(std::unique_ptr<Node> left,
                                          std::unique_ptr<Node> right) {
    return Node::MakeBinary(NodeType::Multiplication, std::move(left),
                            std::move(right));
  }

  inline static std::unique_ptr<Node> Div(std::unique_ptr<Node> left,
                                          std::unique_ptr<Node> right) {
    return Node::MakeBinary(NodeType::Division, std::move(left),
                            std::move(right));
  }

  void SetRoot(std::unique_ptr<Node> node) { m_Root = std::move(node); }

private:
  std::unique_ptr<Node> m_Root = nullptr;
};
}; // namespace logos