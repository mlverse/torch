#include "torch_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
std::int64_t cpp_torch_reduction_mean () {
  return torch::Reduction::Mean;
}

// [[Rcpp::export]]
std::int64_t cpp_torch_reduction_none () {
  return torch::Reduction::None;
}

// [[Rcpp::export]]
std::int64_t cpp_torch_reduction_sum () {
  return torch::Reduction::Sum;
}