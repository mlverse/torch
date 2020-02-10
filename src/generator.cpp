#include "torch_types.h"
#include "utils.hpp"

// https://pytorch.org/docs/stable/torch.html#generators
// https://github.com/pytorch/pytorch/blob/f531815526c69f432e46fadece44f5d3a9b70e30/torch/csrc/Generator.cpp

// [[Rcpp::export]]
Rcpp::XPtr<torch::Generator *> cpp_torch_generator () {
  auto out = new at::CPUGenerator();
  return make_xptr<torch::Generator *>(out);
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_generator_current_seed (Rcpp::XPtr<torch::Generator *> generator) {
  Rcpp::NumericVector out(1);
  uint64_t seed = (*generator)->current_seed();
  std::memcpy(&(out[0]), &(seed), sizeof(double));
  out.attr("class") = "integer64";
  return out;
}

// [[Rcpp::export]]
void cpp_generator_set_current_seed (Rcpp::XPtr<torch::Generator *> generator, std::uint64_t seed) {
  (*generator)->set_current_seed(seed);
}





