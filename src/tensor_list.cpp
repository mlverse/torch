#include "torchr_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensorList> cpp_torch_tensor_list(const Rcpp::List &x)
{
  void * out = lantern_TensorList();

  for (int i = 0; i < x.length(); i++)
  {
    lantern_TensorList_push_back(out, Rcpp::as<Rcpp::XPtr<XPtrTorch>>(x[i])->get());
  }

  return make_xptr<XPtrTorchTensorList>(out);
}

// [[Rcpp::export]]
Rcpp::List cpp_tensor_list_to_r_list(Rcpp::XPtr<XPtrTorchTensorList> x)
{

  Rcpp::List out;
  int64_t sze = lantern_TensorList_size(x->get());

  for (int i = 0; i < sze; i++)
  {
    void * tmp = lantern_TensorList_at(x->get(), i);
    out.push_back(make_xptr<XPtrTorchTensor>(tmp));
  }

  return out;
}
