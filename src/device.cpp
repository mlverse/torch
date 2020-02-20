#include "lantern/lantern.h"
#include "torch_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
std::string cpp_device_type_to_string (Rcpp::XPtr<void*> device) {
  return std::string(lanternDeviceType(device));
}

// [[Rcpp::export]]
std::int64_t cpp_device_index_to_int (Rcpp::XPtr<void*> device) {
  return lanternDeviceIndex(device);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_device (std::string type, Rcpp::Nullable<std::int64_t> index) {
  int64_t index64 = index.isNull() ? 0 : Rcpp::as<std::int64_t>(index);
  XPtrTorch* device = new XPtrTorch(lanternDevice(type.c_str(), index64, !index.isNull()));
  
  return Rcpp::XPtr<XPtrTorch>(device);
}
