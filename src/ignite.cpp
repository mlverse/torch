#include "torch_types.h"
#include <torch.h>

// [[Rcpp::export]]
optim_adamw rcpp_ignite_adamw (torch::TensorList params, double lr, double beta1, double beta2, double eps, double weight_decay, bool amsgrad) {
  return  ignite_adamw(params.get(), lr, beta1, beta2, eps, weight_decay, amsgrad);
}

// [[Rcpp::export]]
optim_param_groups rcpp_ignite_adamw_get_param_groups (optim_adamw groups) {
  return ignite_adamw_get_param_groups(groups.get());
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
void rcpp_ignite_adamw_set_states (optim_adamw opt, torch::TensorList states) {
   ignite_adamw_set_states(opt.get(), states.get());
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
Rcpp::List rcpp_as_list_adamw_param_groups (optim_param_groups groups) {
  auto x = groups.get();
  int size = rcpp_ignite_adamw_param_groups_size(x);
  Rcpp::List lst = Rcpp::List::create();
  for (int i = 0; i < size; i++) {
    Rcpp::List lst_inner = Rcpp::List::create();
    auto params = rcpp_ignite_optim_get_param_group_params(groups, i);
    auto y = ignite_adamw_get_param_group_options(x, i);;
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