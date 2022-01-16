#include <torch.h>

// [[Rcpp::export]]
std::int64_t cpp_torch_reduction_mean() { return lantern_Reduction_Mean(); }

// [[Rcpp::export]]
std::int64_t cpp_torch_reduction_none() { return lantern_Reduction_None(); }

// [[Rcpp::export]]
std::int64_t cpp_torch_reduction_sum() { return lantern_Reduction_Sum(); }
