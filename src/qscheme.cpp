#include "torch_types.h"
#include "utils.hpp" 

// [[Rcpp::export]]
std::string cpp_qscheme_to_string(Rcpp::XPtr<XPtrTorch> x) {
  return lantern_QScheme_type(x->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_per_channel_affine () {
  return make_xptr<XPtrTorch>(lantern_QScheme_per_channel_affine());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_per_tensor_affine () {
  return make_xptr<XPtrTorch>(lantern_QScheme_per_tensor_affine());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_per_channel_symmetric () {
  return make_xptr<XPtrTorch>(lantern_QScheme_per_channel_symmetric());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_per_tensor_symmetric () {
  return make_xptr<XPtrTorch>(lantern_QScheme_per_tensor_symmetric());
}

