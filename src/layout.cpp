#include <torch.h>

// [[Rcpp::export]]
std::string cpp_layout_to_string(Rcpp::XPtr<XPtrTorchLayout> layout_ptr) {
  return std::string(lantern_Layout_string(layout_ptr->get()));
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchLayout> cpp_torch_strided() {
  return make_xptr<XPtrTorchLayout>(lantern_Layout_strided());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchLayout> cpp_torch_sparse() {
  return make_xptr<XPtrTorchLayout>(lantern_Layout_sparse());
}
