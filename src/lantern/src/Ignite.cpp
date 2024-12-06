#include <torch/optim/adamw.h>
#define LANTERN_BUILD
#include <torch/torch.h>
#include "lantern/lantern.h"
#include "utils.hpp"

using optim_adamw = torch::optim::AdamW*;

void* _ignite_adamw(void* params, double lr, double beta1, double beta2,
                        double eps, double weight_decay, bool amsgrad) {
  auto params_ = from_raw::TensorList(params);

  auto options = torch::optim::AdamWOptions(lr)
    .betas(std::make_tuple(beta1, beta2))
    .eps(eps)
    .weight_decay(weight_decay)
    .amsgrad(amsgrad);

  return (void*) new torch::optim::AdamW(params_.vec(), options);
}

void* _ignite_adamw_get_param_groups(void* groups) {
  auto optim = reinterpret_cast<torch::optim::AdamW*>(groups);
  auto param_groups = optim->param_groups();
  return new std::vector<torch::optim::OptimizerParamGroup>(param_groups);
}

int _ignite_adamw_param_groups_size(void* groups) {
  // this is the same method for all optimizers
  auto param_groups = reinterpret_cast<std::vector<torch::optim::OptimizerParamGroup>*>(groups);
  return param_groups->size();
}

void* _ignite_optim_get_param_group_params(void* groups, int i) {
  auto param_groups = reinterpret_cast<std::vector<torch::optim::OptimizerParamGroup>*>(groups);
  auto params = param_groups->at(i).params();
  std::cout << "params size: " << params.size() << std::endl;
  auto out= make_raw::TensorList(params); 
  std::cout << "out created" << std::endl;
  return out;
}

// We don't need to call this from R as it is just used on the Rcpp side
// [[torch::export(rcpp = FALSE, register_types=list(c("adamw_options", "AdamWOptions", "adamw_options", "adamw_options")))]]
adamw_options _ignite_adamw_get_param_group_options(void* groups, int i) {
  auto param_groups = reinterpret_cast<std::vector<torch::optim::OptimizerParamGroup>*>(groups);
  auto options = param_groups->at(i).options();
  auto& x = static_cast<torch::optim::AdamWOptions&>(options);

  auto betas = x.betas();
  adamw_options opts;
  opts.lr = x.lr();
  opts.weight_decay = x.weight_decay();
  opts.betas[0] = std::get<0>(betas);
  opts.betas[1] = std::get<1>(betas);
  opts.eps = x.eps();
  opts.amsgrad = x.amsgrad();
  return opts;
}

// [[torch::export(rcpp = FALSE)]]
void _ignite_adamw_set_param_group_options(void* opt, int i, adamw_options options) {
  auto optim = reinterpret_cast<torch::optim::AdamW*>(opt);
  auto& group = (optim->param_groups())[i];
  auto& options_ref = group.options();
  auto& x = reinterpret_cast<torch::optim::AdamWOptions&>(options_ref);
  x.lr(options.lr);
  x.weight_decay(options.weight_decay);
  x.betas(std::make_tuple(options.betas[0], options.betas[1]));
  x.eps(options.eps);
  x.amsgrad(options.amsgrad);
  return;
}

void* _ignite_adamw_get_states(void* optim) {
  auto opt = reinterpret_cast<torch::optim::AdamW*>(optim);
  // here we iterate over the param states in the order of param groups
  // and cast those param states to the optimizer-specific type
  // the states themselves are still owned by the optimizer itself
  // however, the newly heap-allocated vector is owned by this function
  // so we will have to delete it later

  // Create an empty vector to store the AdamW parameter states

  std::vector<torch::Tensor> tensors;

  // Iterate through each parameter group in the optimizer
  for (const auto& group : opt->param_groups()) {
    // For each parameter in the group
    for (const auto& param : group.params()) {
      
      // Look up this parameter's state in the optimizer's state map
      // The state map is a flat_hash_map that maps parameter keys to their optimizer states
      auto state_it = opt->state().find(param.unsafeGetTensorImpl());

      if (state_it != opt->state().end()) {
        // If state exists for this parameter:
        // 1. Get raw pointer to the OptimizerParamState from the unique_ptr
        // 2. Cast it to AdamWParamState since we know this is an AdamW optimizer
        // 3. Store the pointer in our states vector
        auto base_state = state_it->second.get(); // Get raw pointer from unique_ptr
        auto adamw_state = static_cast<torch::optim::AdamWParamState*>(base_state);
        // we need to clone because the tensors are behind unique pointers
        // but we want ownership
        tensors.push_back(adamw_state->exp_avg().clone());
        tensors.push_back(adamw_state->exp_avg_sq().clone());
        tensors.push_back(adamw_state->max_exp_avg_sq().clone());
        tensors.push_back(torch::scalar_tensor(adamw_state->step(), torch::kLong));
      } else {
        // This parameter should have state - error if not found
        throw std::runtime_error("State not found for parameter");
      }
    }
  }
  std::cout << "tensors size: " << tensors.size() << std::endl;
  return make_raw::TensorList(tensors);
}

void _ignite_adamw_set_states(void* optim, void* states_) {
  auto opt = reinterpret_cast<torch::optim::AdamW*>(optim);
  auto states = from_raw::TensorList(states_);
  size_t i = 0;
  for (const auto& group : opt->param_groups()) {
    for (const auto& param : group.params()) {

      auto state_it = opt->state().find(param.unsafeGetTensorImpl());
      // TODO: Check whether this actually does what we want
      if (state_it != opt->state().end()) {
        auto* current_state = static_cast<torch::optim::AdamWParamState*>(state_it->second.get());
        current_state->exp_avg(states[i]);
        current_state->exp_avg_sq(states[i + 1]);
        current_state->max_exp_avg_sq(states[i + 2]);
        auto step = states[i + 3];
        // convert step from torch::kLong to int64_t
        current_state->step(step.item<int64_t>());
      } else {
        // runtime error
        throw std::runtime_error("State not found");
      }
    }
    i += 4;
  }
}


// [[torch::export(register_types=list(c("script_module", "ScriptModule", "void*", "Rcpp::XPtr<XPtrTorchScriptModule>"), c("torch_stack", "TorchStack", "void*", "XPtrTorchStack"), c("optim", "Optim", "void*", "ignite::optim")))]]
// std::vector<torch::Tensor> ignite_opt_step(script_module network, script_module loss_fn, torch_stack input, torch::Tensor target, optim_adamw optimizer) {
//   // TODO: optim_adamw -> optim
//   optimizer->zero_grad();

//   auto out = (*network)(*input);
//   auto loss_inputs = new torch::jit::Stack();
//   loss_inputs->push_back(out);
//   loss_inputs->push_back(target);

//   auto loss = (*loss_fn)(*loss_inputs);
//   loss.toTensor().backward();
//   optimizer->step();

//   std::vector<torch::Tensor> result;
//   result.push_back(loss.toTensor());
//   result.push_back(out.toTensor());
//   return result;
// }

void _ignite_adamw_step(void* optim) {
  auto opt = reinterpret_cast<torch::optim::AdamW*>(optim);
  opt->step();
}

void _ignite_adamw_zero_grad(void* optim) {
  auto opt = reinterpret_cast<torch::optim::AdamW*>(optim);
  opt->zero_grad();
}
