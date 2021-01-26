#include "torch_types.h"
#include "utils.h"

XPtrTorchTensor::operator SEXP () const {
  auto xptr = make_xptr<XPtrTorchTensor>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_tensor", "R7");
  return xptr; 
}

XPtrTorchTensorList::operator SEXP () const {
  Rcpp::List out;
  int64_t sze = lantern_TensorList_size(this->get());
  
  for (int i = 0; i < sze; i++)
  {
    void * tmp = lantern_TensorList_at(this->get(), i);
    out.push_back(XPtrTorchTensor(tmp));
  }
  
  return out;
}

XPtrTorchScalarType::operator SEXP () const {
  auto xptr = make_xptr<XPtrTorchScalarType>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_dtype", "R7");
  return xptr; 
}

// [[Rcpp::export]]
[[gnu::noinline]]
XPtrTorchTensor test_fun (Rcpp::XPtr<XPtrTorchTensor> x)
{
  return *x;
}