#include "torchr_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
std::string cpp_memory_format_to_string(Rcpp::XPtr<XPtrTorchMemoryFormat> x) {
  return lantern_MemoryFormat_type(x.get()->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchMemoryFormat> cpp_torch_contiguous_format () {
  return make_xptr<XPtrTorchMemoryFormat>(lantern_MemoryFormat_Contiguous());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchMemoryFormat> cpp_torch_preserve_format () {
  return make_xptr<XPtrTorchMemoryFormat>(lantern_MemoryFormat_Preserve());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchMemoryFormat> cpp_torch_channels_last_format () {
  return make_xptr<XPtrTorchMemoryFormat>(lantern_MemoryFormat_ChannelsLast());
}
