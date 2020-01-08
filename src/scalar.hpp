#include "torch_types.h"

template<class T>
torch::Scalar scalar_from_r_impl_ (const SEXP x);

torch::Scalar scalar_from_r_ (SEXP x);

torch::ScalarType scalar_type_from_string(std::string scalar_type);
torch::ScalarType scalar_type_from_string(Rcpp::Nullable<std::string> scalar_type);

std::string scalar_type_to_string(torch::ScalarType scalar_type);

std::string caffe_type_to_string (caffe2::TypeMeta type);

SEXP scalar_to_r_ (torch::Scalar x);
