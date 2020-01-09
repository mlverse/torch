#include "torch_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
std::string cpp_dtype_to_string(Rcpp::XPtr<torch::Dtype> dtype_ptr) {
  
  torch::Dtype dtype = * dtype_ptr;
  
  if (dtype == torch::kFloat32) {
    return "float32";
  }
  
  if (dtype == torch::kFloat64) {
    return "float64"; 
  }
  
  if (dtype == torch::kFloat16) {
    return "float16";
  }
  
  if (dtype == torch::kUInt8) {
    return "uint8";
  }
  
  if (dtype == torch::kInt8) {
    return "int8";
  }
  
  if (dtype == torch::kInt16) {
    return "int16";
  }
  
  if (dtype == torch::kInt32) {
    return "int32";
  }
  
  if (dtype == torch::kInt64) {
    return "int64";
  }
  
  if (dtype == torch::kBool) {
    return "bool";
  }
  
  Rcpp::stop("dtype not handled.");
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Dtype> cpp_torch_float32 () {
  return make_xptr<torch::Dtype>(torch::kFloat32);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Dtype> cpp_torch_float64 () {
  return make_xptr<torch::Dtype>(torch::kFloat64);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Dtype> cpp_torch_float16 () {
  return make_xptr<torch::Dtype>(torch::kFloat16);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Dtype> cpp_torch_uint8 () {
  return make_xptr<torch::Dtype>(torch::kUInt8);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Dtype> cpp_torch_int8 () {
  return make_xptr<torch::Dtype>(torch::kInt8);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Dtype> cpp_torch_int16 () {
  return make_xptr<torch::Dtype>(torch::kInt16);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Dtype> cpp_torch_int32 () {
  return make_xptr<torch::Dtype>(torch::kInt32);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Dtype> cpp_torch_int64 () {
  return make_xptr<torch::Dtype>(torch::kInt64);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Dtype> cpp_torch_bool () {
  return make_xptr<torch::Dtype>(torch::kBool);
}








