#include <torch.h>

// [[Rcpp::export]]
XPtrTorchTensorOptions cpp_torch_tensor_options(
    Rcpp::Nullable<Rcpp::XPtr<XPtrTorchDtype>> dtype_ptr,
    Rcpp::Nullable<Rcpp::XPtr<XPtrTorch>> layout_ptr,
    Rcpp::Nullable<Rcpp::XPtr<XPtrTorch>> device_ptr,
    Rcpp::Nullable<bool> requires_grad, Rcpp::Nullable<bool> pinned_memory) {
  XPtrTorchTensorOptions options(lantern_TensorOptions());

  if (dtype_ptr.isNotNull()) {
    auto dtype = *Rcpp::as<Rcpp::XPtr<XPtrTorch>>(dtype_ptr);
    options = lantern_TensorOptions_dtype(options.get(), dtype.get());
  }

  if (layout_ptr.isNotNull()) {
    auto layout = *Rcpp::as<Rcpp::XPtr<XPtrTorch>>(layout_ptr);
    options = lantern_TensorOptions_layout(options.get(), layout.get());
  }

  if (device_ptr.isNotNull()) {
    auto device = *Rcpp::as<Rcpp::XPtr<XPtrTorch>>(device_ptr);
    options = lantern_TensorOptions_device(options.get(), device.get());
  }

  if (requires_grad.isNotNull()) {
    options = lantern_TensorOptions_requires_grad(
        options.get(), Rcpp::as<bool>(requires_grad));
  }

  if (pinned_memory.isNotNull()) {
    options = lantern_TensorOptions_pinned_memory(
        options.get(), Rcpp::as<bool>(pinned_memory));
  }

  return XPtrTorchTensorOptions(options);
}

// [[Rcpp::export]]
void cpp_torch_tensor_options_print(Rcpp::XPtr<XPtrTorchTensorOptions> x) {
  lantern_TensorOptions_print(x->get());
}
