#include "torch_types.h"
#include "scalar.hpp"

Rcpp::XPtr<torch::Tensor> make_tensor_ptr (torch::Tensor x) {
  auto * out = new torch::Tensor(x);
  return Rcpp::XPtr<torch::Tensor>(out);
}

Rcpp::XPtr<torch::ScalarType> make_scalar_type_ptr (torch::ScalarType x) {
  auto * out = new torch::ScalarType(x);
  return Rcpp::XPtr<torch::ScalarType>(out);
}

Rcpp::List tensorlist_to_r (torch::TensorList x) {
  Rcpp::List out;

  for (int i = 0; i < x.size(); i ++) {
    auto tmp = make_tensor_ptr(x.at(i));
    out.push_back(tmp);
  }

  return out;
}

Rcpp::XPtr<torch::QScheme> make_qscheme_ptr (torch::QScheme x) {
  auto * out = new torch::QScheme(x);
  return Rcpp::XPtr<torch::QScheme>(out);
}

std::vector<torch::Tensor> tensor_list_from_r_ (Rcpp::List x) {
  std::vector<torch::Tensor> out;
  for (int i = 0; i < x.size(); i++) {
    auto tmp = Rcpp::as<Rcpp::XPtr<torch::Tensor>>(x[i]);
    out.push_back(*tmp);
  }
  return out;
}

std::vector<torch::Tensor> tensor_list_from_r_(Rcpp::Nullable<Rcpp::List> x) {
  if (x.isNull()) {
    return {};
  } else {
    return tensor_list_from_r_(Rcpp::as<Rcpp::List>(x));
  }
}

torch::optional<std::int64_t> resolve_null_argument (Rcpp::Nullable<std::int64_t> x) {
  if (x.isNull()) {
    return torch::nullopt;
  } else {
    return Rcpp::as<std::int64_t>(x);
  }
}

torch::optional<bool> resolve_null_argument(Rcpp::Nullable<bool> x) {
  if (x.isNull()) {
    return torch::nullopt;
  } else {
    return Rcpp::as<bool>(x);
  }
}

torch::optional<torch::Scalar> resolve_null_scalar (SEXP x) {
  if (Rf_isNull(x)) {
    return torch::nullopt;
  } else {
    return scalar_from_r_(x);
  }
}

torch::optional<torch::Dimname> resolve_null_argument (Rcpp::Nullable<Rcpp::XPtr<torch::Dimname>> x) {
  if (Rf_isNull(x)) {
    return torch::nullopt;
  } else {
    return * Rcpp::as<Rcpp::XPtr<torch::Dimname>>(x);
  }
}



