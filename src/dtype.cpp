#include "torch_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
std::string cpp_dtype_to_string(Rcpp::XPtr<XPtrTorch> dtype) {
  return lantern_Dtype_type(dtype.get()->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_float32 () {
  return make_xptr<XPtrTorch>(lantern_Dtype_float32());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_float64 () {
  return make_xptr<XPtrTorch>(lantern_Dtype_float64());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_float16 () {
  return make_xptr<XPtrTorch>(lantern_Dtype_float16());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_uint8 () {
  return make_xptr<XPtrTorch>(lantern_Dtype_uint8());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_int8 () {
  return make_xptr<XPtrTorch>(lantern_Dtype_int8());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_int16 () {
  return make_xptr<XPtrTorch>(lantern_Dtype_int16());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_int32 () {
  return make_xptr<XPtrTorch>(lantern_Dtype_int32());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_int64 () {
  return make_xptr<XPtrTorch>(lantern_Dtype_int64());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_bool () {
  return make_xptr<XPtrTorch>(lantern_Dtype_bool());
}