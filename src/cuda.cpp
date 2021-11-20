#include "torch_types.h"
#include "utils.h"

// [[Rcpp::export]]
bool cpp_cuda_is_available () {
  return lantern_cuda_is_available();
}

// [[Rcpp::export]]
int cpp_cuda_device_count () {
  return lantern_cuda_device_count();
}

// [[Rcpp::export]]
int64_t cpp_cuda_current_device() {
  return lantern_cuda_current_device();
}

// [[Rcpp::export]]
XPtrTorchvector_int64_t cpp_cuda_get_device_capability(int64_t device) {
  return lantern_cuda_get_device_capability(device);
}
