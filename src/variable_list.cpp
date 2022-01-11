#include <torch.h>

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchvariable_list> cpp_torch_variable_list(
    const Rcpp::List &x) {
  XPtrTorchvariable_list out = lantern_variable_list_new();

  for (int i = 0; i < x.length(); i++) {
    lantern_variable_list_push_back(
        out.get(), Rcpp::as<Rcpp::XPtr<XPtrTorch>>(x[i])->get());
  }

  return make_xptr<XPtrTorchvariable_list>(out);
}

// [[Rcpp::export]]
Rcpp::List cpp_variable_list_to_r_list(Rcpp::XPtr<XPtrTorchvariable_list> x) {
  Rcpp::List out;
  int64_t sze = lantern_variable_list_size(x->get());

  for (int64_t i = 0; i < sze; i++) {
    void *tmp = lantern_variable_list_get(x->get(), i);
    out.push_back(XPtrTorchTensor(tmp));
  }

  return out;
}
