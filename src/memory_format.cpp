#include "torch_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
std::string cpp_memory_format_to_string(Rcpp::XPtr<torch::MemoryFormat> x) {
  
  torch::MemoryFormat y = * x;
  
  if (y == torch::MemoryFormat::Contiguous) {
    return "contiguous";
  }
  
  if (y == torch::MemoryFormat::Preserve) {
    return "preserve";
  }
  
  if (y == torch::MemoryFormat::ChannelsLast) {
    return "channels_last";
  }
  
  Rcpp::stop("MemoryFormat not handled.");
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::MemoryFormat> cpp_torch_contiguous_format () {
  return make_xptr<torch::MemoryFormat>(torch::MemoryFormat::Contiguous);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::MemoryFormat> cpp_torch_preserve_format () {
  return make_xptr<torch::MemoryFormat>(torch::MemoryFormat::Preserve);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::MemoryFormat> cpp_torch_channels_last_format () {
  return make_xptr<torch::MemoryFormat>(torch::MemoryFormat::ChannelsLast);
}



