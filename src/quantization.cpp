#include "torch_types.h"
#include "utils.hpp"


// [[Rcpp::export]]
bool cpp_Tensor_is_quantized (Rcpp::XPtr<XPtrTorchTensor> self) {
  return lantern_Tensor_is_quantized(self->get());
}

