#include "torch_api.h"
#include "torch_types.h"
#include <torch.h>

// [[Rcpp::export]]
optim_param_groups rcpp_ignite_optim_get_param_groups (optim_adamw opt) {
  return ignite_optim_get_param_groups(opt.get());
}

// [[Rcpp::export]]
int rcpp_ignite_optim_param_groups_size (optim_param_groups groups) {
  return ignite_optim_param_groups_size(groups.get());
}

// [[Rcpp::export]]
torch::TensorList rcpp_ignite_optim_get_param_group_params (optim_param_groups groups, int i) {
  return ignite_optim_get_param_group_params(groups.get(), i);
}

// [[Rcpp::export]]
void rcpp_ignite_optim_step (optim_adamw opt) {
   ignite_optim_step(opt.get());
}
// [[Rcpp::export]]
void rcpp_ignite_optim_zero_grad (optim_adamw opt) {
   ignite_optim_zero_grad(opt.get());
}

// [[Rcpp::export]]
torch::TensorList rcpp_ignite_optim_parameters_with_state(optim_adamw opt) {
  return ignite_adamw_parameters_with_state(opt.get());
}

// [[Rcpp::export]]
optim_adamw rcpp_ignite_adamw (torch::TensorList params, double lr, Rcpp::DoubleVector betas, double eps, double weight_decay, bool amsgrad) {
  double beta1 = betas[0];
  double beta2 = betas[1];
  return ignite_adamw(params.get(), lr, beta1, beta2, eps, weight_decay, amsgrad);
}

// [[Rcpp::export]]
torch::TensorList rcpp_ignite_adamw_get_states (optim_adamw opt) {
  return ignite_adamw_get_states(opt.get());
}

// [[Rcpp::export]]
void rcpp_ignite_adamw_set_states (optim_adamw opt, torch::TensorList params, torch::TensorList states) {
   ignite_adamw_set_states(opt.get(), params.get(), states.get());
}


// [[Rcpp::export]]
void rcpp_ignite_adamw_add_param_group(optim_adamw opt, torch::TensorList params, double lr, Rcpp::DoubleVector betas, double eps, double weight_decay, bool amsgrad) {
  adamw_options options;
  options.lr = lr;
  options.betas[0] = betas[0];
  options.betas[1] = betas[1];
  options.eps = eps;
  options.weight_decay = weight_decay;
  options.amsgrad = amsgrad;
  return ignite_adamw_add_param_group(opt.get(), params.get(), options);
}

// [[Rcpp::export]]
Rcpp::List rcpp_as_list_adamw_param_groups (optim_param_groups groups) {
  int size = rcpp_ignite_optim_param_groups_size(groups);

  Rcpp::List lst = Rcpp::List::create();
  for (int i = 0; i < size; i++) {
    Rcpp::List lst_inner = Rcpp::List::create();
    auto params = rcpp_ignite_optim_get_param_group_params(groups, i);
    auto y = ignite_adamw_get_param_group_options(groups.get(), i);
    lst_inner["params"] = params;
    lst_inner["lr"] = y.lr;
    lst_inner["weight_decay"] = y.weight_decay;
    lst_inner["betas"] = Rcpp::NumericVector::create(y.betas[0], y.betas[1]);
    lst_inner["eps"] = y.eps;
    lst_inner["amsgrad"] = y.amsgrad;

    lst.push_back(lst_inner);
  }
  return lst;
}

// [[Rcpp::export]]
void rcpp_ignite_adamw_set_param_group_options(optim_adamw opt, Rcpp::List list) {
  for (int i = 0; i < list.length(); i++) {
    Rcpp::List group_options = list[i];
    adamw_options opts;
    opts.lr = group_options["lr"];

    Rcpp::NumericVector betas = group_options["betas"];
    opts.betas[0] = betas[0];
    opts.betas[1] = betas[1];
    opts.eps = group_options["eps"];
    opts.weight_decay = group_options["weight_decay"];
    opts.amsgrad = group_options["amsgrad"];
    ignite_adamw_set_param_group_options(opt.get(), i, opts);
  }
}

// [[Rcpp::export]]
optim_adam rcpp_ignite_adam (torch::TensorList params, double lr, Rcpp::DoubleVector betas, double eps, double weight_decay, bool amsgrad) {
  double beta1 = betas[0];
  double beta2 = betas[1];
  return ignite_adam(params.get(), lr, beta1, beta2, eps, weight_decay, amsgrad);
}

// [[Rcpp::export]]
torch::TensorList rcpp_ignite_adam_get_states (optim_adam opt) {
  return ignite_adam_get_states(opt.get());
}

// [[Rcpp::export]]
void rcpp_ignite_adam_set_states (optim_adam opt, torch::TensorList params, torch::TensorList states) {
   ignite_adam_set_states(opt.get(), params.get(), states.get());
}

// [[Rcpp::export]]
void rcpp_ignite_adam_add_param_group(optim_adam opt, torch::TensorList params, double lr, Rcpp::DoubleVector betas, double eps, double weight_decay, bool amsgrad) {
  adam_options options;
  options.lr = lr;
  options.betas[0] = betas[0];
  options.betas[1] = betas[1];
  options.eps = eps;
  options.weight_decay = weight_decay;
  options.amsgrad = amsgrad;
  return ignite_adam_add_param_group(opt.get(), params.get(), options);
}

// [[Rcpp::export]]
Rcpp::List rcpp_as_list_adam_param_groups (optim_param_groups groups) {
  int size = rcpp_ignite_optim_param_groups_size(groups);

  Rcpp::List lst = Rcpp::List::create();
  for (int i = 0; i < size; i++) {
    Rcpp::List lst_inner = Rcpp::List::create();
    auto params = rcpp_ignite_optim_get_param_group_params(groups, i);
    auto y = ignite_adam_get_param_group_options(groups.get(), i);
    lst_inner["params"] = params;
    lst_inner["lr"] = y.lr;
    lst_inner["weight_decay"] = y.weight_decay;
    lst_inner["betas"] = Rcpp::NumericVector::create(y.betas[0], y.betas[1]);
    lst_inner["eps"] = y.eps;
    lst_inner["amsgrad"] = y.amsgrad;

    lst.push_back(lst_inner);
  }
  return lst;
}

// [[Rcpp::export]]
void rcpp_ignite_adam_set_param_group_options(optim_adam opt, Rcpp::List list) {
  for (int i = 0; i < list.length(); i++) {
    Rcpp::List group_options = list[i];
    adam_options opts;
    opts.lr = group_options["lr"];

    Rcpp::NumericVector betas = group_options["betas"];
    opts.betas[0] = betas[0];
    opts.betas[1] = betas[1];
    opts.eps = group_options["eps"];
    opts.weight_decay = group_options["weight_decay"];
    opts.amsgrad = group_options["amsgrad"];
    ignite_adam_set_param_group_options(opt.get(), i, opts);
  }
}

// [[Rcpp::export]]
optim_sgd rcpp_ignite_sgd (torch::TensorList params, double lr, double momentum, double dampening, double weight_decay, bool nesterov) {
  return ignite_sgd(params.get(), lr, momentum, dampening, weight_decay, nesterov);
}

// [[Rcpp::export]]
torch::TensorList rcpp_ignite_sgd_get_states (optim_sgd opt) {
  return ignite_sgd_get_states(opt.get());
}

// [[Rcpp::export]]
void rcpp_ignite_sgd_set_states (optim_sgd opt, torch::TensorList params, torch::TensorList states) {
   ignite_sgd_set_states(opt.get(), params.get(), states.get());
}

// [[Rcpp::export]]
void rcpp_ignite_sgd_add_param_group(optim_sgd opt, torch::TensorList params, double lr, double momentum, double dampening, double weight_decay, bool nesterov) {
  sgd_options options;
  options.lr = lr;
  options.momentum = momentum;
  options.dampening = dampening;
  options.weight_decay = weight_decay;
  options.nesterov = nesterov;
  return ignite_sgd_add_param_group(opt.get(), params.get(), options);
}

// [[Rcpp::export]]
Rcpp::List rcpp_as_list_sgd_param_groups (optim_param_groups groups) {
  int size = rcpp_ignite_optim_param_groups_size(groups);

  Rcpp::List lst = Rcpp::List::create();
  for (int i = 0; i < size; i++) {
    Rcpp::List lst_inner = Rcpp::List::create();
    auto params = rcpp_ignite_optim_get_param_group_params(groups, i);
    auto y = ignite_sgd_get_param_group_options(groups.get(), i);
    lst_inner["params"] = params;
    lst_inner["lr"] = y.lr;
    lst_inner["weight_decay"] = y.weight_decay;
    lst_inner["momentum"] = y.momentum;
    lst_inner["dampening"] = y.dampening;
    lst_inner["nesterov"] = y.nesterov;

    lst.push_back(lst_inner);
  }
  return lst;
}

// [[Rcpp::export]]
void rcpp_ignite_sgd_set_param_group_options(optim_sgd opt, Rcpp::List list) {
  for (int i = 0; i < list.length(); i++) {
    Rcpp::List group_options = list[i];
    sgd_options opts;
    opts.lr = group_options["lr"];

    opts.momentum = group_options["momentum"];
    opts.dampening = group_options["dampening"];
    opts.weight_decay = group_options["weight_decay"];
    opts.nesterov = group_options["nesterov"];
    ignite_sgd_set_param_group_options(opt.get(), i, opts);
  }
}

// [[Rcpp::export]]
optim_rmsprop rcpp_ignite_rmsprop (torch::TensorList params, double lr, double alpha, double eps, double weight_decay, double momentum, bool centered) {
  return ignite_rmsprop(params.get(), lr, alpha, eps, weight_decay, momentum, centered);
}

// [[Rcpp::export]]
torch::TensorList rcpp_ignite_rmsprop_get_states (optim_rmsprop opt) {
  return ignite_rmsprop_get_states(opt.get());
}

// [[Rcpp::export]]
void rcpp_ignite_rmsprop_set_states (optim_rmsprop opt, torch::TensorList params, torch::TensorList states) {
   ignite_rmsprop_set_states(opt.get(), params.get(), states.get());
}

// [[Rcpp::export]]
void rcpp_ignite_rmsprop_add_param_group(optim_rmsprop opt, torch::TensorList params, double lr, double alpha, double eps, double weight_decay, double momentum, bool centered) {
  rmsprop_options options;
  options.lr = lr;
  options.alpha = alpha;
  options.eps = eps;
  options.weight_decay = weight_decay;
  options.momentum = momentum;
  options.centered = centered;
  return ignite_rmsprop_add_param_group(opt.get(), params.get(), options);
}

// [[Rcpp::export]]
Rcpp::List rcpp_as_list_rmsprop_param_groups (optim_param_groups groups) {
  int size = rcpp_ignite_optim_param_groups_size(groups);

  Rcpp::List lst = Rcpp::List::create();
  for (int i = 0; i < size; i++) {
    Rcpp::List lst_inner = Rcpp::List::create();
    auto params = rcpp_ignite_optim_get_param_group_params(groups, i);
    auto y = ignite_rmsprop_get_param_group_options(groups.get(), i);
    lst_inner["params"] = params;
    lst_inner["lr"] = y.lr;
    lst_inner["weight_decay"] = y.weight_decay;
    lst_inner["alpha"] = y.alpha;
    lst_inner["eps"] = y.eps;
    lst_inner["momentum"] = y.momentum;
    lst_inner["centered"] = y.centered;

    lst.push_back(lst_inner);
  }
  return lst;
}

// [[Rcpp::export]]
void rcpp_ignite_rmsprop_set_param_group_options(optim_rmsprop opt, Rcpp::List list) {
  for (int i = 0; i < list.length(); i++) {
    Rcpp::List group_options = list[i];
    rmsprop_options opts;
    opts.lr = group_options["lr"];

    opts.alpha = group_options["alpha"];
    opts.eps = group_options["eps"];
    opts.weight_decay = group_options["weight_decay"];
    opts.momentum = group_options["momentum"];
    opts.centered = group_options["centered"];
    ignite_rmsprop_set_param_group_options(opt.get(), i, opts);
  }
}

// [[Rcpp::export]]
optim_adagrad rcpp_ignite_adagrad (torch::TensorList params, double lr, double lr_decay, double weight_decay, double eps, double initial_accumulator_value) {
  return ignite_adagrad(params.get(), lr, lr_decay, weight_decay, eps, initial_accumulator_value);
}

// [[Rcpp::export]]
torch::TensorList rcpp_ignite_adagrad_get_states (optim_adagrad opt) {
  return ignite_adagrad_get_states(opt.get());
}

// [[Rcpp::export]]
void rcpp_ignite_adagrad_set_states (optim_adagrad opt, torch::TensorList params, torch::TensorList states) {
   ignite_adagrad_set_states(opt.get(), params.get(), states.get());
}

// [[Rcpp::export]]
void rcpp_ignite_adagrad_add_param_group(optim_adagrad opt, torch::TensorList params, double lr, double lr_decay, double weight_decay, double eps, double initial_accumulator_value) {
  adagrad_options options;
  options.lr = lr;
  options.lr_decay = lr_decay;
  options.eps = eps;
  options.weight_decay = weight_decay;
  options.initial_accumulator_value = initial_accumulator_value;
  return ignite_adagrad_add_param_group(opt.get(), params.get(), options);
}

// [[Rcpp::export]]
Rcpp::List rcpp_as_list_adagrad_param_groups (optim_param_groups groups) {
  int size = rcpp_ignite_optim_param_groups_size(groups);

  Rcpp::List lst = Rcpp::List::create();
  for (int i = 0; i < size; i++) {
    Rcpp::List lst_inner = Rcpp::List::create();
    auto params = rcpp_ignite_optim_get_param_group_params(groups, i);
    auto y = ignite_adagrad_get_param_group_options(groups.get(), i);
    lst_inner["params"] = params;
    lst_inner["lr"] = y.lr;
    lst_inner["weight_decay"] = y.weight_decay;
    lst_inner["lr_decay"] = y.lr_decay;
    lst_inner["eps"] = y.eps;
    lst_inner["initial_accumulator_value"] = y.initial_accumulator_value;

    lst.push_back(lst_inner);
  }
  return lst;
}

// [[Rcpp::export]]
void rcpp_ignite_adagrad_set_param_group_options(optim_adagrad opt, Rcpp::List list) {
  for (int i = 0; i < list.length(); i++) {
    Rcpp::List group_options = list[i];
    adagrad_options opts;
    opts.lr = group_options["lr"];

    opts.lr_decay = group_options["lr_decay"];
    opts.eps = group_options["eps"];
    opts.weight_decay = group_options["weight_decay"];
    opts.initial_accumulator_value = group_options["initial_accumulator_value"];
    ignite_adagrad_set_param_group_options(opt.get(), i, opts);
  }
}
