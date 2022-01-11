#include <torch.h>

// [[Rcpp::export]]
std::string cpp_qscheme_to_string(Rcpp::XPtr<XPtrTorchQScheme> x) {
  return lantern_QScheme_type(x->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchQScheme> cpp_torch_per_channel_affine() {
  return make_xptr<XPtrTorchQScheme>(lantern_QScheme_per_channel_affine());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchQScheme> cpp_torch_per_tensor_affine() {
  return make_xptr<XPtrTorchQScheme>(lantern_QScheme_per_tensor_affine());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchQScheme> cpp_torch_per_channel_symmetric() {
  return make_xptr<XPtrTorchQScheme>(lantern_QScheme_per_channel_symmetric());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchQScheme> cpp_torch_per_tensor_symmetric() {
  return make_xptr<XPtrTorchQScheme>(lantern_QScheme_per_tensor_symmetric());
}
