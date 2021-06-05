#include "torch_types.h"
#include "utils.h"

// [[Rcpp::export]]
XPtrTorchTensor cpp_contrib_torch_sparsemax (Rcpp::XPtr<XPtrTorchTensor> input, int dim)
{
  XPtrTorchTensor out = lantern_contrib_torch_sparsemax(input->get(), dim);
  return out;
}
