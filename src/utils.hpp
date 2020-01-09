#include "torch_types.h"

template <class type>
Rcpp::XPtr<type> make_xptr  (type x) {
  auto * out = new type(x);
  return Rcpp::XPtr<type>(out);
}

Rcpp::XPtr<torch::Tensor> make_tensor_ptr (torch::Tensor x);

Rcpp::XPtr<torch::ScalarType> make_scalar_type_ptr (torch::ScalarType x);

Rcpp::List tensorlist_to_r (torch::TensorList x);

Rcpp::XPtr<torch::QScheme> make_qscheme_ptr (torch::QScheme x);

std::vector<torch::Tensor> tensor_list_from_r_ (Rcpp::List x);

std::vector<torch::Tensor> tensor_list_from_r_(Rcpp::Nullable<Rcpp::List> x);

torch::optional<std::int64_t> resolve_null_argument (Rcpp::Nullable<std::int64_t> x);

torch::optional<bool> resolve_null_argument(Rcpp::Nullable<bool> x);

torch::optional<torch::Dimname> resolve_null_argument (Rcpp::Nullable<Rcpp::XPtr<torch::Dimname>> x);

torch::optional<torch::Scalar> resolve_null_scalar (SEXP x);

template<int N>
std::array<bool, N> vector_to_array_bool (std::vector<bool> x) {
  // https://stackoverflow.com/questions/21276889/copy-stdvector-into-stdarray
  std::array<bool, N> arr;
  std::copy_n(x.begin(), N, arr.begin());
  return arr;
}


