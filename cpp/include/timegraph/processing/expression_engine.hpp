#pragma once
/**
 * @file expression_engine.hpp
 * @brief Streaming expression evaluator for calculated parameters
 *
 * Evaluates mathematical expressions on time-series data with streaming
 * (chunk-by-chunk) processing. Supports:
 * - Binary ops: +, -, *, /, **, %
 * - Unary ops: -, abs, sqrt, sin, cos, tan, log, log10, exp, floor, ceil
 * - Multi-arg: min, max, clip, where
 */

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace timegraph {
namespace expr {

// Forward declarations
class ExpressionNode;
using ExprPtr = std::shared_ptr<ExpressionNode>;

/**
 * @brief Types of expression nodes
 */
enum class NodeType { COLUMN, CONSTANT, BINARY_OP, UNARY_OP, FUNCTION };

/**
 * @brief Context for expression evaluation
 *
 * Provides column data access during streaming evaluation.
 */
class EvaluationContext {
public:
  using ColumnProvider =
      std::function<std::vector<double>(const std::string &, size_t, size_t)>;

  EvaluationContext() = default;

  // Set a column provider callback for streaming data access
  void set_column_provider(ColumnProvider provider) {
    column_provider_ = std::move(provider);
  }

  // Get column data for a range
  std::vector<double> get_column(const std::string &name, size_t start,
                                 size_t count) const {
    if (column_provider_) {
      return column_provider_(name, start, count);
    }
    return {};
  }

  // Set preloaded column data (for batch evaluation)
  void set_column_data(const std::string &name,
                       const std::vector<double> &data) {
    preloaded_columns_[name] = data;
  }

  // Get preloaded column data
  const std::vector<double> *
  get_preloaded_column(const std::string &name) const {
    auto it = preloaded_columns_.find(name);
    return it != preloaded_columns_.end() ? &it->second : nullptr;
  }

private:
  ColumnProvider column_provider_;
  std::unordered_map<std::string, std::vector<double>> preloaded_columns_;
};

/**
 * @brief Base class for expression nodes
 */
class ExpressionNode {
public:
  virtual ~ExpressionNode() = default;

  virtual NodeType node_type() const = 0;
  virtual std::vector<double> evaluate(const EvaluationContext &ctx,
                                       size_t length) const = 0;
  virtual std::vector<std::string> get_dependencies() const = 0;
  virtual std::string to_string() const = 0;
};

/**
 * @brief Column reference node
 */
class ColumnNode : public ExpressionNode {
public:
  explicit ColumnNode(std::string name) : name_(std::move(name)) {}

  NodeType node_type() const override { return NodeType::COLUMN; }

  std::vector<double> evaluate(const EvaluationContext &ctx,
                               size_t length) const override {
    // Try preloaded first
    if (auto *data = ctx.get_preloaded_column(name_)) {
      return *data;
    }
    // Otherwise use provider
    return ctx.get_column(name_, 0, length);
  }

  std::vector<std::string> get_dependencies() const override { return {name_}; }

  std::string to_string() const override { return name_; }

  const std::string &name() const { return name_; }

private:
  std::string name_;
};

/**
 * @brief Constant value node
 */
class ConstantNode : public ExpressionNode {
public:
  explicit ConstantNode(double value) : value_(value) {}

  NodeType node_type() const override { return NodeType::CONSTANT; }

  std::vector<double> evaluate(const EvaluationContext &ctx,
                               size_t length) const override {
    return std::vector<double>(length, value_);
  }

  std::vector<std::string> get_dependencies() const override { return {}; }

  std::string to_string() const override { return std::to_string(value_); }

  double value() const { return value_; }

private:
  double value_;
};

/**
 * @brief Binary operation node
 */
class BinaryOpNode : public ExpressionNode {
public:
  enum class Op { ADD, SUB, MUL, DIV, POW, MOD };

  BinaryOpNode(ExprPtr left, ExprPtr right, Op op)
      : left_(std::move(left)), right_(std::move(right)), op_(op) {}

  NodeType node_type() const override { return NodeType::BINARY_OP; }

  std::vector<double> evaluate(const EvaluationContext &ctx,
                               size_t length) const override {
    auto left_vals = left_->evaluate(ctx, length);
    auto right_vals = right_->evaluate(ctx, length);
    std::vector<double> result(length);

    for (size_t i = 0; i < length; ++i) {
      double l = left_vals.size() > i ? left_vals[i] : 0.0;
      double r = right_vals.size() > i ? right_vals[i] : 0.0;

      switch (op_) {
      case Op::ADD:
        result[i] = l + r;
        break;
      case Op::SUB:
        result[i] = l - r;
        break;
      case Op::MUL:
        result[i] = l * r;
        break;
      case Op::DIV:
        result[i] = r != 0.0 ? l / r : 0.0;
        break;
      case Op::POW:
        result[i] = std::pow(l, r);
        break;
      case Op::MOD:
        result[i] = r != 0.0 ? std::fmod(l, r) : 0.0;
        break;
      }
    }
    return result;
  }

  std::vector<std::string> get_dependencies() const override {
    auto deps = left_->get_dependencies();
    auto right_deps = right_->get_dependencies();
    deps.insert(deps.end(), right_deps.begin(), right_deps.end());
    return deps;
  }

  std::string to_string() const override {
    static const char *op_strs[] = {"+", "-", "*", "/", "**", "%"};
    return "(" + left_->to_string() + " " + op_strs[static_cast<int>(op_)] +
           " " + right_->to_string() + ")";
  }

private:
  ExprPtr left_, right_;
  Op op_;
};

/**
 * @brief Unary operation node
 */
class UnaryOpNode : public ExpressionNode {
public:
  enum class Op {
    NEG,
    ABS,
    SQRT,
    SIN,
    COS,
    TAN,
    LOG,
    LOG10,
    EXP,
    FLOOR,
    CEIL,
    ROUND
  };

  UnaryOpNode(ExprPtr operand, Op op) : operand_(std::move(operand)), op_(op) {}

  NodeType node_type() const override { return NodeType::UNARY_OP; }

  std::vector<double> evaluate(const EvaluationContext &ctx,
                               size_t length) const override {
    auto vals = operand_->evaluate(ctx, length);
    std::vector<double> result(vals.size());

    for (size_t i = 0; i < vals.size(); ++i) {
      double v = vals[i];
      switch (op_) {
      case Op::NEG:
        result[i] = -v;
        break;
      case Op::ABS:
        result[i] = std::abs(v);
        break;
      case Op::SQRT:
        result[i] = std::sqrt(v);
        break;
      case Op::SIN:
        result[i] = std::sin(v);
        break;
      case Op::COS:
        result[i] = std::cos(v);
        break;
      case Op::TAN:
        result[i] = std::tan(v);
        break;
      case Op::LOG:
        result[i] = std::log(v);
        break;
      case Op::LOG10:
        result[i] = std::log10(v);
        break;
      case Op::EXP:
        result[i] = std::exp(v);
        break;
      case Op::FLOOR:
        result[i] = std::floor(v);
        break;
      case Op::CEIL:
        result[i] = std::ceil(v);
        break;
      case Op::ROUND:
        result[i] = std::round(v);
        break;
      }
    }
    return result;
  }

  std::vector<std::string> get_dependencies() const override {
    return operand_->get_dependencies();
  }

  std::string to_string() const override {
    static const char *op_strs[] = {"-",   "abs",   "sqrt", "sin",
                                    "cos", "tan",   "log",  "log10",
                                    "exp", "floor", "ceil", "round"};
    return std::string(op_strs[static_cast<int>(op_)]) + "(" +
           operand_->to_string() + ")";
  }

private:
  ExprPtr operand_;
  Op op_;
};

/**
 * @brief Expression Engine for streaming evaluation
 */
class ExpressionEngine {
public:
  ExpressionEngine() = default;

  // Register an expression with a name
  void register_expression(const std::string &name, ExprPtr expr) {
    expressions_[name] = std::move(expr);
  }

  // Evaluate expression by name
  std::vector<double> evaluate(const std::string &name,
                               const EvaluationContext &ctx,
                               size_t length) const {
    auto it = expressions_.find(name);
    if (it == expressions_.end()) {
      return {};
    }
    return it->second->evaluate(ctx, length);
  }

  // Get dependencies for an expression
  std::vector<std::string> get_dependencies(const std::string &name) const {
    auto it = expressions_.find(name);
    if (it == expressions_.end()) {
      return {};
    }
    return it->second->get_dependencies();
  }

  // Check if expression exists
  bool has_expression(const std::string &name) const {
    return expressions_.find(name) != expressions_.end();
  }

  // Get all expression names
  std::vector<std::string> get_expression_names() const {
    std::vector<std::string> names;
    names.reserve(expressions_.size());
    for (auto it = expressions_.begin(); it != expressions_.end(); ++it) {
      names.push_back(it->first);
    }
    return names;
  }

private:
  std::unordered_map<std::string, ExprPtr> expressions_;
};

// Factory functions for creating nodes (used by Python bindings)
inline ExprPtr make_column(const std::string &name) {
  return std::make_shared<ColumnNode>(name);
}

inline ExprPtr make_constant(double value) {
  return std::make_shared<ConstantNode>(value);
}

inline ExprPtr make_binary(ExprPtr left, ExprPtr right, const std::string &op) {
  BinaryOpNode::Op op_enum;
  if (op == "+")
    op_enum = BinaryOpNode::Op::ADD;
  else if (op == "-")
    op_enum = BinaryOpNode::Op::SUB;
  else if (op == "*")
    op_enum = BinaryOpNode::Op::MUL;
  else if (op == "/")
    op_enum = BinaryOpNode::Op::DIV;
  else if (op == "**" || op == "^")
    op_enum = BinaryOpNode::Op::POW;
  else if (op == "%")
    op_enum = BinaryOpNode::Op::MOD;
  else
    return nullptr;

  return std::make_shared<BinaryOpNode>(std::move(left), std::move(right),
                                        op_enum);
}

inline ExprPtr make_unary(ExprPtr operand, const std::string &op) {
  UnaryOpNode::Op op_enum;
  if (op == "-" || op == "neg")
    op_enum = UnaryOpNode::Op::NEG;
  else if (op == "abs")
    op_enum = UnaryOpNode::Op::ABS;
  else if (op == "sqrt")
    op_enum = UnaryOpNode::Op::SQRT;
  else if (op == "sin")
    op_enum = UnaryOpNode::Op::SIN;
  else if (op == "cos")
    op_enum = UnaryOpNode::Op::COS;
  else if (op == "tan")
    op_enum = UnaryOpNode::Op::TAN;
  else if (op == "log")
    op_enum = UnaryOpNode::Op::LOG;
  else if (op == "log10")
    op_enum = UnaryOpNode::Op::LOG10;
  else if (op == "exp")
    op_enum = UnaryOpNode::Op::EXP;
  else if (op == "floor")
    op_enum = UnaryOpNode::Op::FLOOR;
  else if (op == "ceil")
    op_enum = UnaryOpNode::Op::CEIL;
  else if (op == "round")
    op_enum = UnaryOpNode::Op::ROUND;
  else
    return nullptr;

  return std::make_shared<UnaryOpNode>(std::move(operand), op_enum);
}

} // namespace expr
} // namespace timegraph
