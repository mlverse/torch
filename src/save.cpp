#include "torch_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
std::string cpp_tensor_save (Rcpp::XPtr<XPtrTorchTensor> x)
{
  const char * s = lantern_tensor_save(x->get());
  return std::string(s);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensor> cpp_tensor_load (std::string s)
{
  XPtrTorchTensor t = lantern_tensor_load(s.c_str());
  return make_xptr<XPtrTorchTensor>(t);
}