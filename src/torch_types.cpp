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

XPtrTorchScalar::operator SEXP () const {
  XPtrTorchScalarType dtype_ptr = lantern_Scalar_dtype(this->get());
  const char * dtype_c = lantern_Dtype_type(dtype_ptr.get());
  auto dtype = std::string(dtype_c);
  lantern_const_char_delete(dtype_c);
  
  Rcpp::RObject output;
  if (dtype == "Double") {
    output = lantern_Scalar_to_double(this->get());
  } else if (dtype == "Float") {
    output = lantern_Scalar_to_float(this->get());
  } else if (dtype == "Bool") {
    output = lantern_Scalar_to_bool(this->get());
  } else if (dtype == "Int") {
    output = lantern_Scalar_to_int(this->get());
  } else if (dtype == "Long") {
    output = lantern_Scalar_to_int(this->get());
  } else {
    Rcpp::stop("Cannot convert from scalar of type.");
  }
  
  return output; 
}

// [[Rcpp::export]]
[[gnu::noinline]]
XPtrTorchTensor test_fun (Rcpp::XPtr<XPtrTorchTensor> x)
{
  return *x;
}