#include "torch_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchScalar> cpp_torch_scalar (SEXP x) {

  XPtrTorchScalar out;
  std::string type;
  
  int i;
  double d;
  bool b;
  
  switch (TYPEOF(x)) {
  case INTSXP:
    i = Rcpp::as<int>(x); 
    type = "int";
    out = lantern_Scalar((void*)(&i), type.c_str()); 
    break;
  case REALSXP:
    d = Rcpp::as<double>(x); 
    type = "double";
    out = lantern_Scalar((void*)(&d), type.c_str()); 
    break;
  case LGLSXP:
    b = Rcpp::as<bool>(x); 
    type = "bool";
    out = lantern_Scalar((void*)(&b), type.c_str()); 
    break;
  case CHARSXP:
    Rcpp::stop("strings are not handled yet");
  default:
    Rcpp::stop("not handled");
  }

  return make_xptr<XPtrTorchScalar>(out);
}