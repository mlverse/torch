/*
#include "torch_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
Rcpp::XPtr<torch::Scalar> cpp_torch_scalar (SEXP x) {
  torch::Scalar out;
  
  switch (TYPEOF(x)) {
  case INTSXP:
    out = Rcpp::as<int>(x);
    break;
  case REALSXP:
    out = Rcpp::as<double>(x);
    break;
  case LGLSXP:
    out = Rcpp::as<bool>(x);
    break;
  case CHARSXP:
    Rcpp::stop("strings are not handled yet");
  default:
    Rcpp::stop("not handled");
  }
  
  return make_xptr<torch::Scalar>(out);
}*/