#include "torch_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
Rcpp::XPtr<std::vector<torch::Tensor>> cpp_torch_tensor_list (const Rcpp::List & x) {
  std::vector<torch::Tensor> out;
  
  for (int i = 0; i < x.length(); i++) {
    torch::Tensor tmp = *Rcpp::as<Rcpp::XPtr<torch::Tensor>>(x[i]);
    out.push_back(tmp);
  }
  
  return make_xptr<std::vector<torch::Tensor>>(out);
}

// [[Rcpp::export]]
Rcpp::List cpp_tensor_list_to_r_list (Rcpp::XPtr<std::vector<torch::Tensor>> x) {
  
  Rcpp::List out;
  auto y = *x;
  
  for (int i = 0; i < x->size(); i ++) {
    auto tmp = make_xptr<torch::Tensor>(y.at(i));
    out.push_back(tmp);
  }
  
  return out;
}