#include "lantern/lantern.h"
#include "torch_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
std::string cpp_device_type_to_string(Rcpp::XPtr<XPtrTorchDevice> device)
{
  return std::string(lantern_Device_type(device->get()));
}

// [[Rcpp::export]]
std::int64_t cpp_device_index_to_int(Rcpp::XPtr<XPtrTorchDevice> device)
{
  return lantern_Device_index(device->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchDevice> cpp_torch_device(std::string type, Rcpp::Nullable<std::int64_t> index)
{
  int64_t index64 = index.isNull() ? 0 : Rcpp::as<std::int64_t>(index);
  XPtrTorchDevice device = lantern_Device(type.c_str(), index64, !index.isNull());

  return make_xptr<XPtrTorchDevice>(device);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchOptionalDeviceGuard> cpp_optional_device_guard (Rcpp::XPtr<XPtrTorchDevice> device)
{
  Rcpp::Rcout << "hey!" << std::endl;
  void* dg = lantern_OptionalDeviceGuard_set_device(device->get());
  return make_xptr<XPtrTorchOptionalDeviceGuard>(dg);
}

