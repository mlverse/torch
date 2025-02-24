#include <ostream>
#include <torch/optim/adamw.h>
#include <torch/types.h>
#define LANTERN_BUILD
#include <torch/torch.h>
#include "lantern/lantern.h"
#include "utils.hpp"

using optim_adagrad = torch::optim::Adagrad*;
using optim_adam = torch::optim::Adam*;
using optim_adamw = torch::optim::AdamW*;
using optim_rmsprop = torch::optim::RMSprop*;
using optim_sgd = torch::optim::SGD*;

// functions that are common to all optimizers

void* _ignite_optim_get_param_groups(void* opt) {
  auto optim = reinterpret_cast<torch::optim::Optimizer*>(opt);
  auto param_groups = optim->param_groups();
  return (void*) new std::vector<torch::optim::OptimizerParamGroup>(param_groups);
}

int _ignite_optim_param_groups_size(void* groups) {
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

void* _ignite_optim_parameters_with_state(void* optim) {
  auto opt = reinterpret_cast<torch::optim::Optimizer*>(optim);
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

void _ignite_optim_step(void* optim) {
  auto opt = reinterpret_cast<torch::optim::Optimizer*>(optim);
  opt->step();
}

void _ignite_optim_zero_grad(void* optim) {
  auto opt = reinterpret_cast<torch::optim::Optimizer*>(optim);
  opt->zero_grad();
}

// adagrad

torch::optim::AdagradOptions _ignite_adagrad_options(double lr, double lr_decay, double weight_decay, double eps, double initial_accumulator_value) {
  return torch::optim::AdagradOptions(lr)
    .lr_decay(lr_decay)
    .weight_decay(weight_decay)
    .initial_accumulator_value(initial_accumulator_value)
    .eps(eps);
}

void* _ignite_adagrad(void* params, double lr, double lr_decay, double weight_decay, double eps, double initial_accumulator_value) {
  auto params_ = from_raw::TensorList(params);
  auto options = _ignite_adagrad_options(lr, lr_decay, weight_decay, eps, initial_accumulator_value);
  return (void*) new torch::optim::Adagrad(params_.vec(), options);
}

void _ignite_adagrad_add_param_group(void* optim, void* params, adagrad_options options) {
  auto optim_ = reinterpret_cast<torch::optim::Adagrad*>(optim);
  auto params_ = from_raw::TensorList(params);
  auto options_ = _ignite_adagrad_options(options.lr, options.lr_decay, options.weight_decay, options.eps, options.initial_accumulator_value);

  auto options_ptr = std::make_unique<torch::optim::AdagradOptions>(options_);
  auto group = torch::optim::OptimizerParamGroup(params_.vec(), std::move(options_ptr));
  optim_->add_param_group(group);
}

adagrad_options _ignite_adagrad_get_param_group_options(void* groups, int i) {
  auto param_groups = reinterpret_cast<std::vector<torch::optim::OptimizerParamGroup>*>(groups);
  auto g = (*param_groups)[i];
  auto& x = static_cast<torch::optim::AdagradOptions&>(g.options());

  adagrad_options opts { x.lr(), x.lr_decay(), x.weight_decay(), x.initial_accumulator_value(), x.eps() };

  return opts;
}

void _ignite_adagrad_set_param_group_options(void* opt, int i, adagrad_options options) {
  auto optim = reinterpret_cast<torch::optim::Adagrad*>(opt);
  auto& group = (optim->param_groups())[i];
  auto& options_ref = group.options();
  auto& x = reinterpret_cast<torch::optim::AdagradOptions&>(options_ref);
  x.lr(options.lr)
    .lr_decay(options.lr_decay)
    .weight_decay(options.weight_decay)
    .initial_accumulator_value(options.initial_accumulator_value)
    .eps(options.eps);
}

void* _ignite_adagrad_get_states(void* optim) {
  auto opt = reinterpret_cast<torch::optim::Adagrad*>(optim);
  std::vector<torch::Tensor> tensors;
  for (const auto& group : opt->param_groups()) {
    for (const auto& param : group.params()) {
      auto state_it = opt->state().find(param.unsafeGetTensorImpl());
      if (state_it != opt->state().end()) { 
        auto base_state = state_it->second.get();
        auto adagrad_state = static_cast<torch::optim::AdagradParamState*>(base_state);
        tensors.push_back(adagrad_state->sum().clone());
        tensors.push_back(torch::tensor({adagrad_state->step()}, torch::kLong));
      }
    }
  }
  return make_raw::TensorList(tensors);
}

void _ignite_adagrad_set_states(void* optim, void* params, void* states_) {
  auto opt = reinterpret_cast<torch::optim::Adagrad*>(optim);
  auto states = from_raw::TensorList(states_);
  auto params_ = from_raw::TensorList(params);

  size_t i = 0;
  for (const auto& param : params_) {
    auto state_it = opt->state().find(param.unsafeGetTensorImpl());
    if (state_it == opt->state().end()) {
      auto new_state = std::make_unique<torch::optim::AdagradParamState>();
      opt->state()[param.unsafeGetTensorImpl()] = std::move(new_state);
      state_it = opt->state().find(param.unsafeGetTensorImpl());
    }
    auto* current_state = static_cast<torch::optim::AdagradParamState*>(state_it->second.get());
    current_state->sum(states[i]);
    current_state->step(states[i + 1].item<int64_t>());
    i += 2;
  }
}

// adam

torch::optim::AdamOptions _ignite_adam_options(double lr, double beta1, double beta2, double eps, double weight_decay, bool amsgrad) {
  return torch::optim::AdamOptions(lr)
    .betas(std::make_tuple(beta1, beta2))
    .eps(eps)
    .weight_decay(weight_decay)
    .amsgrad(amsgrad);
}

void* _ignite_adam(void* params, double lr, double beta1, double beta2,
                        double eps, double weight_decay, bool amsgrad) {
  auto params_ = from_raw::TensorList(params);
  auto options = _ignite_adam_options(lr, beta1, beta2, eps, weight_decay, amsgrad);
  return (void*) new torch::optim::Adam(params_.vec(), options);
}

void _ignite_adam_add_param_group(void* optim, void* params, adam_options options) {
  auto optim_ = reinterpret_cast<torch::optim::Adam*>(optim);
  auto params_ = from_raw::TensorList(params);
  auto options_ = _ignite_adam_options(options.lr, options.betas[0], options.betas[1], options.eps, options.weight_decay, options.amsgrad);
  auto options_ptr = std::make_unique<torch::optim::AdamOptions>(options_); 
  auto group = torch::optim::OptimizerParamGroup(params_.vec(), std::move(options_ptr));
  optim_->add_param_group(group);
}

adam_options _ignite_adam_get_param_group_options(void* groups, int i) {
  auto param_groups = reinterpret_cast<std::vector<torch::optim::OptimizerParamGroup>*>(groups);
  auto g = (*param_groups)[i];
  auto& x = static_cast<torch::optim::AdamOptions&>(g.options());

  adam_options opts { x.lr(), x.weight_decay(),  std::get<0>(x.betas()), std::get<1>(x.betas()), x.eps(), x.amsgrad() }  ;
  return opts;
}

void _ignite_adam_set_param_group_options(void* opt, int i, adam_options options) {
  auto optim = reinterpret_cast<torch::optim::Adam*>(opt);
  auto& group = (optim->param_groups())[i];
  auto& options_ref = group.options();
  auto& x = reinterpret_cast<torch::optim::AdamOptions&>(options_ref);
  x.lr(options.lr);
  x.betas(std::make_tuple(options.betas[0], options.betas[1]));
  x.eps(options.eps);
  x.weight_decay(options.weight_decay);
  x.amsgrad(options.amsgrad);
}

void* _ignite_adam_get_states(void* optim) {
  auto opt = reinterpret_cast<torch::optim::Adam*>(optim);
  std::vector<torch::Tensor> tensors;
  for (const auto& group : opt->param_groups()) {
    for (const auto& param : group.params()) {
      auto state_it = opt->state().find(param.unsafeGetTensorImpl());
      if (state_it != opt->state().end()) {
        auto base_state = state_it->second.get(); // Get raw pointer from unique_ptr
        auto adam_state = static_cast<torch::optim::AdamParamState*>(base_state);
        tensors.push_back(adam_state->exp_avg().clone());
        tensors.push_back(adam_state->exp_avg_sq().clone());
        if (adam_state->max_exp_avg_sq().defined()) {
          tensors.push_back(adam_state->max_exp_avg_sq().clone());
        } else {
          tensors.push_back(torch::empty(0, torch::kFloat32));
        }
        tensors.push_back(torch::tensor({adam_state->step()}, torch::kLong));
      }
    }
  }
  return make_raw::TensorList(tensors);
}

void _ignite_adam_set_states(void* optim, void* params,void* states_) {
  auto opt = reinterpret_cast<torch::optim::Adam*>(optim);
  auto states = from_raw::TensorList(states_);
  auto params_ = from_raw::TensorList(params);

  size_t i = 0;
  for (const auto& param : params_) {
    auto state_it = opt->state().find(param.unsafeGetTensorImpl());
    if (state_it == opt->state().end()) {
      auto new_state = std::make_unique<torch::optim::AdamParamState>();
      opt->state()[param.unsafeGetTensorImpl()] = std::move(new_state);
      state_it = opt->state().find(param.unsafeGetTensorImpl());
    }
    auto* current_state = static_cast<torch::optim::AdamParamState*>(state_it->second.get());
    current_state->exp_avg(states[i]);
    current_state->exp_avg_sq(states[i + 1]);
    if (states[i + 2].numel() != 0) {
      current_state->max_exp_avg_sq(states[i + 2]);
    }
    auto step = states[i + 3];
    current_state->step(step.item<int64_t>());
    i += 4;
  }
}

// adamw

torch::optim::AdamWOptions _ignite_adamw_options(double lr, double beta1, double beta2, double eps, double weight_decay, bool amsgrad) {
  return torch::optim::AdamWOptions(lr)
    .betas(std::make_tuple(beta1, beta2))
    .eps(eps)
    .weight_decay(weight_decay)
    .amsgrad(amsgrad);
}

void* _ignite_adamw(void* params, double lr, double beta1, double beta2,
                        double eps, double weight_decay, bool amsgrad) {
  auto params_ = from_raw::TensorList(params);

  auto options = _ignite_adamw_options(lr, beta1, beta2, eps, weight_decay, amsgrad);

  return (void*) new torch::optim::AdamW(params_.vec(), options);
}

void _ignite_adamw_add_param_group(void* optim, void* params, adamw_options options) {
  auto optim_ = reinterpret_cast<torch::optim::AdamW*>(optim);
  auto params_ = from_raw::TensorList(params);
  auto options_ = _ignite_adamw_options(options.lr, options.betas[0], options.betas[1], options.eps, options.weight_decay, options.amsgrad);
  // need to create an OptimizerParamGroup that is then added
  // create a std::unique_ptr for the options
  // create unique ptr for options
  auto options_ptr = std::make_unique<torch::optim::AdamWOptions>(options_);
  auto group = torch::optim::OptimizerParamGroup(params_.vec(), std::move(options_ptr));
  optim_->add_param_group(group);
}

adamw_options _ignite_adamw_get_param_group_options(void* groups, int i) {
  auto param_groups = reinterpret_cast<std::vector<torch::optim::OptimizerParamGroup>*>(groups);
  auto g = (*param_groups)[i];
  auto& x = static_cast<torch::optim::AdamWOptions&>(g.options());

  adamw_options opts { x.lr(), x.weight_decay(), std::get<0>(x.betas()), std::get<1>(x.betas()), x.eps(), x.amsgrad() };
  return opts;
}

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
          tensors.push_back(torch::empty(0, torch::kFloat32));
        }
        tensors.push_back(torch::tensor({adamw_state->step()}, torch::kLong));
      }
    }
  }
  return make_raw::TensorList(tensors);
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

// rmsprop

torch::optim::RMSpropOptions _ignite_rmsprop_options(double lr, double alpha, double eps, double weight_decay, double momentum, bool centered) {
  return torch::optim::RMSpropOptions(lr)
    .alpha(alpha)
    .eps(eps)
    .weight_decay(weight_decay)
    .momentum(momentum)
    .centered(centered);
}

void* _ignite_rmsprop(void* params, double lr, double alpha, double eps, double weight_decay, double momentum, bool centered) {
  auto params_ = from_raw::TensorList(params);
  auto options = _ignite_rmsprop_options(lr, alpha, eps, weight_decay, momentum, centered);
  return (void*) new torch::optim::RMSprop(params_.vec(), options);
}

void _ignite_rmsprop_add_param_group(void* optim, void* params, rmsprop_options options) {
  auto optim_ = reinterpret_cast<torch::optim::RMSprop*>(optim);
  auto params_ = from_raw::TensorList(params);
  auto options_ = _ignite_rmsprop_options(options.lr, options.alpha, options.eps, options.weight_decay, options.momentum, options.centered);
  auto options_ptr = std::make_unique<torch::optim::RMSpropOptions>(options_);
  auto group = torch::optim::OptimizerParamGroup(params_.vec(), std::move(options_ptr));
  optim_->add_param_group(group);
}

rmsprop_options _ignite_rmsprop_get_param_group_options(void* groups, int i) {
  auto param_groups = reinterpret_cast<std::vector<torch::optim::OptimizerParamGroup>*>(groups);
  auto g = (*param_groups)[i];
  auto& x = static_cast<torch::optim::RMSpropOptions&>(g.options());

  rmsprop_options opts { x.lr(), x.alpha(), x.eps(), x.weight_decay(), x.momentum(), x.centered() };
  return opts;
} 

void _ignite_rmsprop_set_param_group_options(void* optim, int i, rmsprop_options options) {
  auto optim_ = reinterpret_cast<torch::optim::RMSprop*>(optim);
  auto& group = (optim_->param_groups())[i];
  auto& options_ref = group.options();
  auto& x = reinterpret_cast<torch::optim::RMSpropOptions&>(options_ref);
  x.lr(options.lr);
  x.alpha(options.alpha);
  x.eps(options.eps);
  x.weight_decay(options.weight_decay);
  x.momentum(options.momentum);
  x.centered(options.centered);
}

void* _ignite_rmsprop_get_states(void* optim) {
  auto opt = reinterpret_cast<torch::optim::RMSprop*>(optim);
  std::vector<torch::Tensor> tensors;
  for (const auto& group : opt->param_groups()) {
    for (const auto& param : group.params()) {
      auto state_it = opt->state().find(param.unsafeGetTensorImpl());
      if (state_it != opt->state().end()) {
        auto base_state = state_it->second.get(); // Get raw pointer from unique_ptr
        auto rmsprop_state = static_cast<torch::optim::RMSpropParamState*>(base_state);
        if (rmsprop_state->grad_avg().defined()) {
          tensors.push_back(rmsprop_state->grad_avg().clone());
        } else {
          tensors.push_back(torch::empty(0, torch::kFloat32));
        }
        tensors.push_back(rmsprop_state->square_avg().clone());
        if (rmsprop_state->momentum_buffer().defined()) {
          tensors.push_back(rmsprop_state->momentum_buffer().clone());
        } else {
          tensors.push_back(torch::empty(0, torch::kFloat32));
        }
        tensors.push_back(torch::tensor({rmsprop_state->step()}, torch::kLong));
      }
    }
  }
  return make_raw::TensorList(tensors);
} 

void _ignite_rmsprop_set_states(void* optim, void* params, void* states_) {
  auto opt = reinterpret_cast<torch::optim::RMSprop*>(optim);
  auto states = from_raw::TensorList(states_);
  auto params_ = from_raw::TensorList(params);

  size_t i = 0;
  for (const auto& param : params_) {
    auto state_it = opt->state().find(param.unsafeGetTensorImpl());
    if (state_it == opt->state().end()) {
      auto new_state = std::make_unique<torch::optim::RMSpropParamState>();
      opt->state()[param.unsafeGetTensorImpl()] = std::move(new_state);
      state_it = opt->state().find(param.unsafeGetTensorImpl());
    }
    auto* current_state = static_cast<torch::optim::RMSpropParamState*>(state_it->second.get());
    if (states[i].numel() != 0) {
      current_state->grad_avg(states[i]);
    }
    current_state->square_avg(states[i + 1]);
    if (states[i + 2].numel() != 0) {
      current_state->momentum_buffer(states[i + 2]);
    }
    auto step = states[i + 3];
    current_state->step(step.item<int64_t>());
    i += 4;
  }
}

// sgd

torch::optim::SGDOptions _ignite_sgd_options(double lr, double momentum, double dampening, double weight_decay, bool nesterov) {
  return torch::optim::SGDOptions(lr)
    .momentum(momentum)
    .dampening(dampening)
    .weight_decay(weight_decay)
    .nesterov(nesterov);
}

void* _ignite_sgd(void* params, double lr, double momentum, double dampening, double weight_decay, bool nesterov) {
  auto params_ = from_raw::TensorList(params);
  auto options = _ignite_sgd_options(lr, momentum, dampening, weight_decay, nesterov);
  return (void*) new torch::optim::SGD(params_.vec(), options);
}

void _ignite_sgd_add_param_group(void* optim, void* params, sgd_options options) {
  auto optim_ = reinterpret_cast<torch::optim::SGD*>(optim);
  auto params_ = from_raw::TensorList(params);
  auto options_ = _ignite_sgd_options(options.lr, options.momentum, options.dampening, options.weight_decay, options.nesterov);

  auto options_ptr = std::make_unique<torch::optim::SGDOptions>(options_);
  auto group = torch::optim::OptimizerParamGroup(params_.vec(), std::move(options_ptr));
  optim_->add_param_group(group);
}

sgd_options _ignite_sgd_get_param_group_options(void* groups, int i) {
  auto param_groups = reinterpret_cast<std::vector<torch::optim::OptimizerParamGroup>*>(groups);
  auto g = (*param_groups)[i];
  auto& x = static_cast<torch::optim::SGDOptions&>(g.options());

  sgd_options opts { x.lr(), x.momentum(), x.dampening(), x.weight_decay(), x.nesterov() };
  return opts;
}

void _ignite_sgd_set_param_group_options(void* optim, int i, sgd_options options) {
  auto optim_ = reinterpret_cast<torch::optim::SGD*>(optim);
  auto& group = (optim_->param_groups())[i];
  auto& options_ref = group.options();
  auto& x = reinterpret_cast<torch::optim::SGDOptions&>(options_ref);
  x.lr(options.lr);
  x.momentum(options.momentum);
  x.dampening(options.dampening);
  x.weight_decay(options.weight_decay);
  x.nesterov(options.nesterov);
}

void* _ignite_sgd_get_states(void* optim) {
  auto opt = reinterpret_cast<torch::optim::SGD*>(optim);
  std::vector<torch::Tensor> tensors;
  for (const auto& group : opt->param_groups()) {
    for (const auto& param : group.params()) {
      auto state_it = opt->state().find(param.unsafeGetTensorImpl());
      if (state_it != opt->state().end()) { 
        auto base_state = state_it->second.get();
        auto sgd_state = static_cast<torch::optim::SGDParamState*>(base_state);
        if (sgd_state->momentum_buffer().defined()) {
          tensors.push_back(sgd_state->momentum_buffer().clone());
        } else {
          tensors.push_back(torch::empty(0, torch::kFloat32));
        }
      }
    }
  }
  return make_raw::TensorList(tensors);
}

void _ignite_sgd_set_states(void* optim, void* params, void* states_) {
  auto opt = reinterpret_cast<torch::optim::SGD*>(optim);
  auto states = from_raw::TensorList(states_);
  auto params_ = from_raw::TensorList(params);

  size_t i = 0;
  for (const auto& param : params_) {
    auto state_it = opt->state().find(param.unsafeGetTensorImpl());
    if (state_it == opt->state().end()) {
      auto new_state = std::make_unique<torch::optim::SGDParamState>();
      opt->state()[param.unsafeGetTensorImpl()] = std::move(new_state);
      state_it = opt->state().find(param.unsafeGetTensorImpl());
    }
    auto* current_state = static_cast<torch::optim::SGDParamState*>(state_it->second.get());
    if (states[i].defined()) {
      current_state->momentum_buffer(states[i]);
    }
    i += 1;
  }
}
