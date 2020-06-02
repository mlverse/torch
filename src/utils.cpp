#include "torch_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
Rcpp::XPtr<std::nullptr_t> cpp_nullptr () {
  return make_xptr<std::nullptr_t>(nullptr);
}

// [[Rcpp::export]]
Rcpp::XPtr<std::nullptr_t> cpp_nullopt () {
  return make_xptr<std::nullptr_t>(nullptr);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensor> cpp_tensor_undefined () {
  return make_xptr<XPtrTorchTensor>(XPtrTorchTensor(lantern_Tensor_undefined()));
}

