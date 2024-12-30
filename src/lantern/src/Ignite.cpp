#include <ostream>
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

void* _ignite_adamw_get_param_groups(void* opt) {
  // TODO: I think we can just cast to optimizer here so we need only one implementation
  auto optim = reinterpret_cast<torch::optim::AdamW*>(opt);
  auto param_groups = optim->param_groups();
  return (void*) new std::vector<torch::optim::OptimizerParamGroup>(param_groups);
}

int _ignite_adamw_param_groups_size(void* groups) {
  // this is the same method for all optimizers
  auto param_groups = reinterpret_cast<std::vector<torch::optim::OptimizerParamGroup>*>(groups);
  return param_groups->size();
}

void* _ignite_optim_get_param_group_params(void* groups, int i) {
  auto param_groups = reinterpret_cast<std::vector<torch::optim::OptimizerParamGroup>*>(groups);
  auto group = param_groups->at(i);
  auto params = group.params();

  auto out= make_raw::TensorList(params);
  return out;
}

void _ignite_adamw_add_param_group(void* optim, void* params, adamw_options options) {
  auto optim_ = reinterpret_cast<torch::optim::AdamW*>(optim);
  auto params_ = from_raw::TensorList(params);
  auto options_ = torch::optim::AdamWOptions(options.lr)
    .betas(std::make_tuple(options.betas[0], options.betas[1]))
    .eps(options.eps)
    .weight_decay(options.weight_decay)
    .amsgrad(options.amsgrad);
  // need to create an OptimizerParamGroup that is then added
  // create a std::unique_ptr for the options
  // create unique ptr for options
  auto options_ptr = std::make_unique<torch::optim::AdamWOptions>(options_);
  auto group = torch::optim::OptimizerParamGroup(params_.vec(), std::move(options_ptr));
  optim_->add_param_group(group);
}

// We don't need to call this from R as it is just used on the Rcpp side
// [[torch::export(rcpp = FALSE, register_types=list(c("adamw_options", "AdamWOptions", "adamw_options", "adamw_options")))]]
adamw_options _ignite_adamw_get_param_group_options(void* groups, int i) {
  auto param_groups = reinterpret_cast<std::vector<torch::optim::OptimizerParamGroup>*>(groups);
  // TODO: Check why -> .at(i) does not work
  auto g = (*param_groups)[i];
  auto& x = static_cast<torch::optim::AdamWOptions&>(g.options());

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

      if (state_it != opt->state().end()) { // TODO: Check what this does exactly
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
        // check if amsgrad is true, then clone, otherwise set to empty tensor
        if (adamw_state->max_exp_avg_sq().defined()) {
          tensors.push_back(adamw_state->max_exp_avg_sq().clone());
        } else {
          tensors.push_back(torch::Tensor());
        }
        tensors.push_back(torch::scalar_tensor(adamw_state->step(), torch::kLong));
      }
    }
  }
  return make_raw::TensorList(tensors);
}

void* _ignite_adamw_parameters_with_state(void* optim) {
  auto opt = reinterpret_cast<torch::optim::AdamW*>(optim);
  auto params_with_states = std::vector<torch::Tensor>();

  for (const auto& group : opt->param_groups()) {
    for (const auto& param : group.params()) {
      if (opt->state().find(param.unsafeGetTensorImpl()) != opt->state().end()) {
        params_with_states.push_back(param);
      }
    }
  }
  return make_raw::TensorList(params_with_states);
}


void _ignite_adamw_set_states(void* optim, void* params,void* states_) {
  auto opt = reinterpret_cast<torch::optim::AdamW*>(optim);
  auto states = from_raw::TensorList(states_);
  auto params_ = from_raw::TensorList(params);

  size_t i = 0;
  for (const auto& param : params_) {
    auto state_it = opt->state().find(param.unsafeGetTensorImpl());
    if (state_it == opt->state().end()) {
      // initialize a new state
      auto new_state = std::make_unique<torch::optim::AdamWParamState>();
      opt->state()[param.unsafeGetTensorImpl()] = std::move(new_state);
      state_it = opt->state().find(param.unsafeGetTensorImpl());
    }
    auto* current_state = static_cast<torch::optim::AdamWParamState*>(state_it->second.get());
    current_state->exp_avg(states[i]);
    current_state->exp_avg_sq(states[i + 1]);
    // is only defined if amsgrad = TRUE
    if (states[i + 2].defined()) {
      current_state->max_exp_avg_sq(states[i + 2]);
    }
    auto step = states[i + 3];
    // convert step from torch::kLong to int64_t
    current_state->step(step.item<int64_t>());
    i += 4;
  }
}


void _ignite_adamw_step(void* optim) {
  auto opt = reinterpret_cast<torch::optim::AdamW*>(optim);
  opt->step();
}

void _ignite_adamw_zero_grad(void* optim) {
  auto opt = reinterpret_cast<torch::optim::AdamW*>(optim);
  opt->zero_grad();
}
