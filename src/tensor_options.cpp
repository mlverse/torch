#include "torch_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
Rcpp::XPtr<torch::TensorOptions> cpp_torch_tensor_options (
    Rcpp::Nullable<Rcpp::XPtr<torch::Dtype>> dtype_ptr,
    Rcpp::Nullable<Rcpp::XPtr<torch::Layout>> layout_ptr,
    Rcpp::Nullable<Rcpp::XPtr<torch::Device>> device_ptr,
    Rcpp::Nullable<bool> requires_grad,
    Rcpp::Nullable<bool> pinned_memory
) {
  
  auto options = torch::TensorOptions();
  
  if (dtype_ptr.isNotNull()) {
    auto dtype = * Rcpp::as<Rcpp::XPtr<torch::Dtype>>(dtype_ptr);
    options = options.dtype(dtype);
  }
  
  if (layout_ptr.isNotNull()) {
    auto layout = * Rcpp::as<Rcpp::XPtr<torch::Layout>>(layout_ptr);
    options = options.layout(layout);
  }
  
  if (device_ptr.isNotNull()) {
    auto device = * Rcpp::as<Rcpp::XPtr<torch::Device>>(device_ptr);
    options = options.device(device);
  }
  
  if (requires_grad.isNotNull()) {
    options = options.requires_grad(Rcpp::as<bool>(requires_grad));
  }
  
  if (pinned_memory.isNotNull()) {
    options = options.pinned_memory(Rcpp::as<bool>(pinned_memory));
  }
  
  return make_xptr<torch::TensorOptions>(options);
}
