#include <torch.h>

// [[Rcpp::export]]
std::string cpp_memory_format_to_string(Rcpp::XPtr<XPtrTorchMemoryFormat> x) {
  return lantern_MemoryFormat_type(x.get()->get());
}

// [[Rcpp::export]]
XPtrTorchMemoryFormat cpp_torch_contiguous_format() {
  return XPtrTorchMemoryFormat(lantern_MemoryFormat_Contiguous());
}

// [[Rcpp::export]]
XPtrTorchMemoryFormat cpp_torch_preserve_format() {
  return XPtrTorchMemoryFormat(lantern_MemoryFormat_Preserve());
}

// [[Rcpp::export]]
XPtrTorchMemoryFormat cpp_torch_channels_last_format() {
  return XPtrTorchMemoryFormat(lantern_MemoryFormat_ChannelsLast());
}
