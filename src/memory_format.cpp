#include "torch_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
std::string cpp_memory_format_to_string(Rcpp::XPtr<XPtrTorch> x) {
  return lantern_MemoryFormat_type(x.get()->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_contiguous_format () {
  return make_xptr<XPtrTorch>(lantern_MemoryFormat_Contiguous());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_preserve_format () {
  return make_xptr<XPtrTorch>(lantern_MemoryFormat_Preserve());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_channels_last_format () {
  return make_xptr<XPtrTorch>(lantern_MemoryFormat_ChannelsLast());
}
