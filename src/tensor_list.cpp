#include "torchr_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_tensor_list(const Rcpp::List &x)
{
  auto out = lantern_TensorList();

  for (int i = 0; i < x.length(); i++)
  {
    lantern_TensorList_push_back(out, Rcpp::as<Rcpp::XPtr<XPtrTorch>>(x[i])->get());
  }

  return make_xptr<XPtrTorch>(out);
}

// [[Rcpp::export]]
Rcpp::List cpp_tensor_list_to_r_list(Rcpp::XPtr<XPtrTorch> x)
{

  Rcpp::List out;
  int64_t sze = lantern_TensorList_size(x->get());

  for (int i = 0; i < sze; i++)
  {
    XPtrTorch tmp = lantern_TensorList_at(x->get(), i);
    out.push_back(make_xptr<XPtrTorch>(tmp));
  }

  return out;
}
