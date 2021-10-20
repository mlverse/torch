#include <torch.h>
#include "utils.h"

// [[Rcpp::export]]
XPtrTorchIValue ivalue_test_function (XPtrTorchIValue x)
{
  return x;
}
