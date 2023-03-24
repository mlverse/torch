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