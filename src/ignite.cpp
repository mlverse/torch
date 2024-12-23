#include "torch_api.h"
#include "torch_types.h"
#include <torch.h>

// [[Rcpp::export]]
optim_adamw rcpp_ignite_adamw (torch::TensorList params, double lr, Rcpp::DoubleVector betas, double eps, double weight_decay, bool amsgrad) {
  double beta1 = betas[0];
  double beta2 = betas[1];
  return  ignite_adamw(params.get(), lr, beta1, beta2, eps, weight_decay, amsgrad);
}

// [[Rcpp::export]]
optim_param_groups rcpp_ignite_adamw_get_param_groups (optim_adamw opt) {
  return ignite_adamw_get_param_groups(opt.get());
}

// [[Rcpp::export]]
int rcpp_ignite_adamw_param_groups_size (optim_param_groups groups) {
  return ignite_adamw_param_groups_size(groups.get());
}

// [[Rcpp::export]]
torch::TensorList rcpp_ignite_optim_get_param_group_params (optim_param_groups groups, int i) {
  return ignite_optim_get_param_group_params(groups.get(), i);
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
void rcpp_ignite_adamw_step (optim_adamw opt) {
   ignite_adamw_step(opt.get());
}
// [[Rcpp::export]]
void rcpp_ignite_adamw_zero_grad (optim_adamw opt) {
   ignite_adamw_zero_grad(opt.get());
}

// [[Rcpp::export]]
torch::TensorList rcpp_ignite_adamw_parameters_with_state(optim_adamw opt) {
  return ignite_adamw_parameters_with_state(opt.get());
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
  int size = rcpp_ignite_adamw_param_groups_size(groups);

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
