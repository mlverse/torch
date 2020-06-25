#include "torch_types.h"
#include "utils.hpp"
#include <algorithm>

// [[Rcpp::export]]
Rcpp::XPtr<std::nullptr_t> cpp_nullptr () {
  return make_xptr<std::nullptr_t>(nullptr);
}

// [[Rcpp::export]]
Rcpp::XPtr<std::nullptr_t> cpp_nullopt () {
  return make_xptr<std::nullptr_t>(nullptr);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchoptional_int64_t> cpp_optional_int64_t (Rcpp::Nullable<int64_t> x)
{
  XPtrTorchoptional_int64_t out = nullptr;
  if (x.isNull()) {
    out = lantern_optional_int64_t(0, true);
  } else {
    out = lantern_optional_int64_t(Rcpp::as<int64_t>(x), false);
  }
  return make_xptr<XPtrTorchoptional_int64_t>(out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensor> cpp_tensor_undefined () {
  return make_xptr<XPtrTorchTensor>(XPtrTorchTensor(lantern_Tensor_undefined()));
}

// [[Rcpp::export]]
std::string cpp_clean_names (std::string x, std::vector<std::string> r)
{
  std::string out = x;
  char replace;
  for (int i = 0; i < r.size(); i ++)
  {
    replace = r[i][0];
    out.erase(std::remove(out.begin(), out.end(), replace), out.end());  
  }
  return out;
}
