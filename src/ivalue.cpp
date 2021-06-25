#include "torch_types.h"
#include "utils.h"

// [[Rcpp::export]]
XPtrTorchIValue ivalue_test_function (XPtrTorchIValue x)
{
  return x;
}

// [[Rcpp::export]]
void test_ivalue (XPtrTorchTensorList x)
{
  XPtrTorchIValue y = lantern_IValue_from_TensorList(x.get());
  std::cout << lantern_IValue_type(y.get()) << std::endl;
}
