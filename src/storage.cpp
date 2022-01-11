#include <torch.h>

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchStorage> cpp_Tensor_storage(
    Rcpp::XPtr<XPtrTorchTensor> self) {
  XPtrTorchStorage out = lantern_Tensor_storage(self->get());
  return make_xptr<XPtrTorchStorage>(out);
}

// [[Rcpp::export]]
bool cpp_Tensor_has_storage(Rcpp::XPtr<XPtrTorchTensor> self) {
  return lantern_Tensor_has_storage(self->get());
}

// [[Rcpp::export]]
std::string cpp_Storage_data_ptr(Rcpp::XPtr<XPtrTorchStorage> self) {
  void* x = lantern_Storage_data_ptr(self->get());
  std::stringstream ss;
  ss << x;
  std::string str = ss.str();
  return str;
}
