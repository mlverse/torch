#include "Function.h"

#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/torch.h>

#include "Autograd.h"
#include "lantern/lantern.h"

#define LANTERN_ERROR_HANDLE               \
  if (lanternLastError() != NULL) {        \
    std::string last = lanternLastError(); \
    lanternLastErrorClear();               \
    throw std::string(last.c_str());       \
  }

inline std::vector<c10::optional<torch::autograd::Variable>> to_optional(
    torch::autograd::variable_list &output) {
  std::vector<c10::optional<torch::autograd::Variable>> result;
  for (auto &v : output) {
    result.push_back(c10::optional<torch::autograd::Variable>(v));
  }
  return result;
}

namespace torch {
namespace autograd {

variable_list LanternFunction::apply(variable_list args, void *forward_,
                                     void *backward_) {
  std::shared_ptr<LanternNode> node(new LanternNode(), deleteNode);

  auto forward = reinterpret_cast<LanternLambdaFunction *>(forward_);
  auto backward = reinterpret_cast<LanternLambdaFunction *>(backward_);

  node->backward_ = std::shared_ptr<LanternLambdaFunction>(backward);

  const size_t num_inputs = args.size();

  node->is_variable_input_.reserve(num_inputs);

  for (size_t i = 0; i < num_inputs; i++) {
    node->is_variable_input_.push_back(true);
  }

  bool is_executable =
      GradMode::is_enabled() && any_variable_requires_grad(args);

  auto next_edges = collect_next_edges(args);

  node->set_ctx_grad_fn(node);
  node->set_next_edges(std::move(next_edges));
  node->clear_input_metadata();

  node->input_info_.reserve(num_inputs);
  for (auto &var : args) {
    node->input_info_.emplace_back(var);
  }

  variable_list outputs;
  {
    AutoGradMode grad_mode(false);
    outputs = (*forward->fn_)(&node->ctx_, args);
    delete forward;
    LANTERN_ERROR_HANDLE
  }

  auto wrapped_outputs = _wrap_outputs(
      args, node->ctx_.get_non_differentiable(), node->ctx_.get_dirty(),
      to_optional(outputs), is_executable ? node : nullptr,
      // TODO: not sure what this function should actually do. Seems to be
      // related to functorch & co. Hopefully it's not used by the currrent
      // code.
      [](variable_list x, variable_list y) { return x; });

  node->output_info_.reserve(wrapped_outputs.size());
  for (auto &output : wrapped_outputs) {
    if (is_executable && output.has_value()) {
      node->output_info_.emplace_back(output.value());
    } else if (is_executable) {
      node->output_info_.emplace_back();
    }
  }

  if (is_executable) {
    node->save_variables_to_ctx();
  }

  // wrapped_outputs will be a variable_list so, convert it to the correct
  // return type. Only Variable and variable_list are accepted as return types.
  return to_output_type<variable_list>(wrapped_outputs);
}

// The logic here is the same as PyNode::apply, so changes to it should be done
// in both the places
variable_list LanternNode::apply(variable_list &&inputs) {
  at::OptionalDeviceGuard _device_guard;

  int num_inputs = inputs.size();
  variable_list backward_inputs;
  backward_inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    if (inputs[i].defined()) {
      backward_inputs.emplace_back(inputs[i]);
    } else {
      backward_inputs.emplace_back(output_info_[i].zeros(_device_guard));
    }
  }

  auto outputs = (*this->backward_->fn_)(&ctx_, backward_inputs);
  LLOG("Checking outputs")
  if (lanternLastError() != NULL) {
    std::string last = lanternLastError();
    LLOG("Checking outputs, exception: %s", last.c_str());
    lanternLastErrorClear();
    throw std::runtime_error(last.c_str());
  }

  int num_forward_inputs = is_variable_input_.size();
  int num_outputs = outputs.size();
  // Returning too many results is ok, but only as long as they're all
  // undefined. Truncate the result vector in that case.
  if (num_outputs > num_forward_inputs) {
    bool all_undef = true;
    for (int i = num_forward_inputs; i < num_outputs; ++i) {
      all_undef &= (!outputs[i].defined());
    }
    if (all_undef) {
      outputs.resize(num_forward_inputs);
      num_outputs = num_forward_inputs;
    }
  }

  if (num_outputs != num_forward_inputs) {
    std::string msg("function ");
    msg += name() + " returned an incorrect number of gradients (expected ";
    msg += c10::to_string(num_forward_inputs) + ", got ";
    msg += c10::to_string(num_outputs) + ")";
    throw std::runtime_error(msg);
  }

  variable_list results;
  results.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    if (!is_variable_input_[i]) {
      if (outputs[i].defined()) {
        std::string msg("function ");
        msg += name() +
               " returned a gradient different that is defined at position ";
        msg += c10::to_string(i + 1) +
               ", but the corresponding forward input was not a Variable";
        throw std::runtime_error(msg);
      }
      continue;
    }
    if (!outputs[i].defined()) {
      auto &info = input_info_[results.size()];
      if (info.requires_grad) {
        results.emplace_back(info.zeros(_device_guard));
      } else {
        results.emplace_back();
      }
    } else {
      results.emplace_back(outputs[i]);
    }
  }
  return results;
}

void LanternNode::release_variables() {
  ctx_.saved_variables_.clear();
  ctx_.has_freed_buffers_ = true;
}

void LanternNode::save_variables_to_ctx() { ctx_.save_variables(); }

void LanternNode::set_ctx_grad_fn(const std::shared_ptr<Node> &node) {
  ctx_.grad_fn_ = node;
}

void LanternAutogradContext::save_for_backward(variable_list to_save) {
  to_save_ = std::move(to_save);
}

// The logic for handling saved variables here is the same as
// python_function.cpp See _save_variables() and unpack_saved_variables()
void LanternAutogradContext::save_variables() {
  saved_variables_.clear();
  auto ptr = grad_fn_.lock();

  for (const auto &var : to_save_) {
    // Allow empty variables to be saved
    if (var.defined()) {
      bool is_output = var.grad_fn().get() == ptr.get();
      saved_variables_.emplace_back(var, is_output);
    } else {
      saved_variables_.emplace_back();
    }
  }
  to_save_.clear();
}

variable_list LanternAutogradContext::get_saved_variables() const {
  TORCH_CHECK(!has_freed_buffers_, ERR_BACKWARD_TWICE);
  variable_list saved;
  saved.reserve(saved_variables_.size());
  auto ptr = grad_fn_.lock();
  TORCH_INTERNAL_ASSERT(ptr);
  for (auto &var : saved_variables_) {
    saved.push_back(var.unpack(ptr));
  }
  return saved;
}

void LanternAutogradContext::mark_dirty(const variable_list &inputs) {
  dirty_inputs_.clear();
  dirty_inputs_.reserve(inputs.size());
  for (auto &var : inputs) {
    dirty_inputs_.insert(var.unsafeGetTensorImpl());
  }
}

void LanternAutogradContext::mark_non_differentiable(
    const variable_list &outputs) {
  non_differentiable_.clear();
  non_differentiable_.reserve(outputs.size());
  for (auto &var : outputs) {
    non_differentiable_.insert(var.unsafeGetTensorImpl());
  }
}

const std::unordered_set<at::TensorImpl *> &LanternAutogradContext::get_dirty()
    const {
  return dirty_inputs_;
}

const std::unordered_set<at::TensorImpl *>
    &LanternAutogradContext::get_non_differentiable() const {
  return non_differentiable_;
}

void LanternAutogradContext::set_arguments(std::vector<std::string> names,
                                           std::vector<bool> needs_grad) {
  this->argument_names_ = names;
  this->argument_needs_grad_ = needs_grad;
};

std::vector<std::string> LanternAutogradContext::get_argument_names() {
  return argument_names_;
};

std::vector<bool> LanternAutogradContext::get_argument_needs_grad() {
  return argument_needs_grad_;
};

void LanternAutogradContext::set_saved_variables_names(
    std::vector<std::string> names) {
  this->saved_variables_names_ = names;
}

std::vector<std::string> LanternAutogradContext::get_saved_variables_names() {
  return this->saved_variables_names_;
}

}  // namespace autograd
}  // namespace torch
