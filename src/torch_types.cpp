#include "torch_types.h"
#include "utils.h"

XPtrTorchTensor::operator SEXP () const {
  auto xptr = make_xptr<XPtrTorchTensor>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_tensor", "R7");
  return xptr; 
}

// [[Rcpp::export]]
[[gnu::noinline]]
XPtrTorchTensor test_fun (Rcpp::XPtr<XPtrTorchTensor> x)
{
  return *x;
}