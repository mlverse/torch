/*
#include "torch_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
std::string cpp_layout_to_string(Rcpp::XPtr<torch::Layout> layout_ptr) {
  
  torch::Layout layout = * layout_ptr;
  
  if (layout == torch::kStrided) {
    return "strided";
  }
  
  if (layout == torch::kSparse) {
    return "sparse_coo";
  }
  
  Rcpp::stop("layout not handled.");
}



// [[Rcpp::export]]
Rcpp::XPtr<torch::Layout> cpp_torch_strided () {
  return make_xptr<torch::Layout>(torch::kStrided);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Layout> cpp_torch_sparse_coo () {
  return make_xptr<torch::Layout>(torch::kSparse);
}




*/