#include "torchr_types.h"
#include "utils.hpp"

// https://pytorch.org/docs/stable/torch.html#generators
// https://github.com/pytorch/pytorch/blob/f531815526c69f432e46fadece44f5d3a9b70e30/torch/csrc/Generator.cpp

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchGenerator> cpp_torch_generator () {
  XPtrTorchGenerator out = lantern_Generator();
  return make_xptr<XPtrTorchGenerator>(out);
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_generator_current_seed (Rcpp::XPtr<XPtrTorchGenerator> generator) {
  Rcpp::NumericVector out(1);
  uint64_t seed = lantern_Generator_current_seed(generator->get());
  std::memcpy(&(out[0]), &(seed), sizeof(double));
  out.attr("class") = "integer64";
  return out;
}

// [[Rcpp::export]]
void cpp_generator_set_current_seed (Rcpp::XPtr<XPtrTorchGenerator> generator, std::uint64_t seed) {
  lantern_Generator_set_current_seed(generator->get(), seed);
}
