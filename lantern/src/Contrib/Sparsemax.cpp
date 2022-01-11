#define LANTERN_BUILD
#include <torch/torch.h>

#include <iostream>
#include <stdexcept>  // std::out_of_range
#include <string>

#include "../utils.hpp"
#include "lantern/lantern.h"

using namespace torch::autograd;

// Inherit from Function
class SparseMaxFunction : public Function<SparseMaxFunction> {
 public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input,
                               int dim) {
    auto input_dim = input.dim();
    if (input_dim <= dim || dim < -input_dim) {
      throw std::out_of_range("Dimension out of range");
    }

    bool needs_reshaping = input_dim > 2;
    auto original_size = input.sizes().vec();

    if (needs_reshaping) {
      // transpose batch and nth dim
      input = input.transpose(0, dim);

      // Flatten all dimensions except nth dim
      input = input.reshape({input.size(0), -1});

      // Transpose flattened dimensions to 0th dim, nth dim to last dim
      input = input.transpose(0, -1);
    }

    // Translate by max for numerical stability
    input = input - std::get<0>(input.max(-1, true)).expand_as(input);

    auto zs = std::get<0>(input.sort(-1, true));
    auto range = torch::arange(1, input.size(-1) + 1);
    range = range.expand_as(input).to(input);

    // Determine sparsity of projection
    auto bound = 1 + range * zs;
    auto is_gt = bound.gt(zs.cumsum(-1)).to(input.dtype());
    auto k = std::get<0>((is_gt * range).max(-1, true));

    // Compute threshold
    auto zs_sparse = is_gt * zs;

    // Compute taus
    auto taus = (zs_sparse.sum(-1, true) - 1) / k;
    taus = taus.expand_as(input);

    auto output = torch::max(torch::zeros_like(input), input - taus);

    // Save context
    ctx->save_for_backward({output});
    ctx->saved_data["needs_reshaping"] = needs_reshaping;
    ctx->saved_data["dim"] = dim;

    if (needs_reshaping) {
      // Tranpose flattened dim to last dim, nth dim to 0th dim
      output = output.transpose(0, 1);

      // Reshape to original size
      output = output.reshape(original_size);

      // Swap batch dim and nth dim
      output = output.transpose(0, dim);
    }

    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto output = saved[0];
    auto grad_output = grad_outputs[0];

    bool needs_reshaping = ctx->saved_data["needs_reshaping"].toBool();
    int dim = ctx->saved_data["dim"].toInt();
    auto original_size = grad_output.sizes().vec();

    if (needs_reshaping) {
      // transpose batch and nth dim
      grad_output = grad_output.transpose(0, dim);

      // Flatten all dimensions except nth dim
      grad_output = grad_output.reshape({grad_output.size(0), -1});

      // Transpose flattened dimensions to 0th dim, nth dim to last dim
      grad_output = grad_output.transpose(0, -1);
    }

    // Compute gradient
    auto nonzeros = torch::ne(output, 0);
    auto num_nonzeros = nonzeros.sum(-1, true);
    auto sum = (grad_output * nonzeros).sum(-1, true) / num_nonzeros;
    auto grad_input = nonzeros * (grad_output - sum.expand_as(grad_output));

    if (needs_reshaping) {
      // Tranpose flattened dim to last dim, nth dim to 0th dim
      grad_input = grad_input.transpose(0, 1);

      // Reshape to original size
      grad_input = grad_input.reshape(original_size);

      // Swap batch dim and nth dim
      grad_input = grad_input.transpose(0, dim);
    }

    auto o = torch::autograd::variable_list(2);
    o[0] = grad_input;

    return o;
  }
};

void *_lantern_contrib_torch_sparsemax(void *input, int dim) {
  LANTERN_FUNCTION_START
  torch::Tensor t = from_raw::Tensor(input);
  torch::Tensor res = SparseMaxFunction::apply(t, dim);
  return make_raw::Tensor(res);
  LANTERN_FUNCTION_END
}