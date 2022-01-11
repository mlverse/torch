#include <torch.h>

// [[Rcpp::export]]
XPtrTorchTensorList cpp_torch_tensor_list(const Rcpp::List &x) {
  XPtrTorchTensorList out = lantern_TensorList();

  SEXP item;
  for (int i = 0; i < x.length(); i++) {
    item = x.at(i);
    lantern_TensorList_push_back(out.get(), XPtrTorchTensor(item).get());
  }

  return out;
}

// [[Rcpp::export]]
Rcpp::List cpp_tensor_list_to_r_list(Rcpp::XPtr<XPtrTorchTensorList> x) {
  Rcpp::List out;
  int64_t sze = lantern_TensorList_size(x->get());

  for (int i = 0; i < sze; i++) {
    void *tmp = lantern_TensorList_at(x->get(), i);
    out.push_back(XPtrTorchTensor(tmp));
  }

  return out;
}
