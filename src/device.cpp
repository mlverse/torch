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


// This is used by the TensorOptions initializer to chose a 
// proper default for the device.
torch::Device default_device = nullptr;

// [[Rcpp::export]]
void cpp_set_default_device (SEXP device) {
  if (TYPEOF(device) == NILSXP) {
    default_device = nullptr;
  } else {
    default_device = Rcpp::as<torch::Device>(device);  
  }
}

// [[Rcpp::export]]
SEXP cpp_get_current_default_device () {
  if (default_device.get()) {
    return Rcpp::wrap(default_device);
  } else {
    return R_NilValue;
  }
}

torch::Device get_current_device () {
  return default_device;
}