#include <torch.h>

// [[Rcpp::export]]
std::string cpp_layout_to_string(XPtrTorchLayout layout_ptr) {
  return std::string(lantern_Layout_string(layout_ptr.get()));
}

// [[Rcpp::export]]
XPtrTorchLayout cpp_torch_strided() {
  return XPtrTorchLayout(lantern_Layout_strided());
}

// [[Rcpp::export]]
XPtrTorchLayout cpp_torch_sparse() {
  return XPtrTorchLayout(lantern_Layout_sparse());
}
