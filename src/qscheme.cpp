#include "torch_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
std::string cpp_qscheme_to_string(Rcpp::XPtr<torch::QScheme> x) {
  
  torch::QScheme y = * x;
  
  if (y == torch::QScheme::PER_CHANNEL_AFFINE) {
    return "per_channel_affine";
  }
  
  if (y == torch::QScheme::PER_TENSOR_AFFINE) {
    return "per_tensor_affine";
  }
  
  if (y == torch::QScheme::PER_CHANNEL_SYMMETRIC) {
    return "per_channel_symmetric";
  }
  
  if (y == torch::QScheme::PER_TENSOR_SYMMETRIC) {
    return "per_tensor_symmetric";
  }
  
  Rcpp::stop("QScheme not handled.");
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::QScheme> cpp_torch_per_channel_affine () {
  return make_xptr<torch::QScheme>(torch::QScheme::PER_CHANNEL_AFFINE);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::QScheme> cpp_torch_per_tensor_affine () {
  return make_xptr<torch::QScheme>(torch::QScheme::PER_TENSOR_AFFINE);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::QScheme> cpp_torch_per_channel_symmetric () {
  return make_xptr<torch::QScheme>(torch::QScheme::PER_CHANNEL_SYMMETRIC);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::QScheme> cpp_torch_per_tensor_symmetric () {
  return make_xptr<torch::QScheme>(torch::QScheme::PER_TENSOR_SYMMETRIC);
}

