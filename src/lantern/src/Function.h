#pragma once
#include <iostream>

#define LANTERN_BUILD

#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

class LanternLambdaFunction;

namespace torch {
namespace autograd {

// Context to save information during forward that can be accessed in backward
struct LanternAutogradContext {
  LanternAutogradContext() = default;
  LanternAutogradContext(const LanternAutogradContext &other) = delete;
  LanternAutogradContext &operator=(const LanternAutogradContext &other) =
      delete;

  // Can be used to save non-variable data for backward()
  ska::flat_hash_map<std::string, at::IValue> saved_data;

  // Saves the list of variables for a future call to backward(). This
  // should be called at most once from inside of forward().
  void save_for_backward(variable_list to_save);
  // Marks variables in the list as modified in an in-place operation. This
  // should be called at most once from inside of forward() and all arguments
  // should be inputs.
  void mark_dirty(const variable_list &inputs);
  // Marks outputs in the list as not requiring gradients. This should be called
  // at most once from inside of forward() and all arguments should be outputs.
  void mark_non_differentiable(const variable_list &outputs);

  // Get the list of variables that were saved in forward using
  // save_for_backward(). Before returning them to the user, a check is made to
  // ensure that they were not modified by any in-place operations.
  variable_list get_saved_variables() const;
  const std::unordered_set<at::TensorImpl *> &get_dirty() const;
  const std::unordered_set<at::TensorImpl *> &get_non_differentiable() const;

  void set_arguments(std::vector<std::string> names,
                     std::vector<bool> needs_grad);
  std::vector<std::string> get_argument_names();
  std::vector<bool> get_argument_needs_grad();
  void set_saved_variables_names(std::vector<std::string> names);
  std::vector<std::string> get_saved_variables_names();

 private:
  std::unordered_set<at::TensorImpl *> non_differentiable_;
  std::unordered_set<at::TensorImpl *> dirty_inputs_;
  std::vector<torch::autograd::SavedVariable> saved_variables_;
  variable_list to_save_;
  std::vector<std::string> argument_names_;
  std::vector<bool> argument_needs_grad_;
  std::vector<std::string> saved_variables_names_;

  // The CppNode in the autograd graph that owns this AutogradContext. We need a
  // weak_ptr to avoid a refcycle. Since grad_fn_ owns this AutogradContext, it
  // will always be alive when we want to use it.
  std::weak_ptr<Node> grad_fn_;
  bool has_freed_buffers_;

  void save_variables();

  friend struct LanternNode;
};

struct LanternFunction {
  static variable_list apply(variable_list args, void *forward, void *backward);
};

struct LanternNode : public Node {
  variable_list apply(variable_list &&inputs) override;
  LanternAutogradContext ctx_;
  std::vector<bool> is_variable_input_;
  std::vector<VariableInfo> input_info_;
  std::vector<VariableInfo> output_info_;
  std::shared_ptr<LanternLambdaFunction> backward_;

  void release_variables() override;

  void set_ctx_grad_fn(const std::shared_ptr<Node> &node);
  void save_variables_to_ctx();
};

}  // namespace autograd
}  // namespace torch