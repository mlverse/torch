#include <lantern/lantern.h>
#include <torch.h>

// [[Rcpp::export]]
std::string cpp_device_type_to_string(Rcpp::XPtr<XPtrTorchDevice> device) {
  auto s = lantern_Device_type(device->get());
  auto out = std::string(s);
  lantern_const_char_delete(s);
  return out;
}

// [[Rcpp::export]]
std::int64_t cpp_device_index_to_int(Rcpp::XPtr<XPtrTorchDevice> device) {
  return lantern_Device_index(device->get());
}

// [[Rcpp::export]]
XPtrTorchDevice cpp_torch_device(std::string type,
                                 Rcpp::Nullable<std::int64_t> index) {
  int64_t index64 = index.isNull() ? 0 : Rcpp::as<std::int64_t>(index);
  XPtrTorchDevice device =
      lantern_Device(type.c_str(), index64, !index.isNull());

  return device;
}
