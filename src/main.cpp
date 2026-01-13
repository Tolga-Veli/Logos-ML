#include <iostream>
#include "Expression-Tree.hpp"

int main() {
  using Tree = logos::ExpressionTree<>;
  auto root = Tree::Mul(Tree::Value(2.0), Tree::Value(2.0));
  Tree t;
  t.SetRoot(std::move(root));
  std::cout << t.Evaluate() << '\n';
}