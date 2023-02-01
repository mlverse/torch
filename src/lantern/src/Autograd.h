#pragma once
#include <torch/torch.h>

#include "utils.hpp"

namespace torch {
namespace autograd {
class LanternAutogradContext;
}
}  // namespace torch

using autograd_fun = std::function<torch::autograd::variable_list(
    torch::autograd::LanternAutogradContext*, torch::autograd::variable_list)>;

class LanternLambdaFunction {
 public:
  std::shared_ptr<autograd_fun> fn_;
  std::shared_ptr<void> rcpp_fn;
  LanternLambdaFunction(autograd_fun fn, void* rcpp_fn);
};