#include "torch_types.h"

// Create scalars from R SEXPS -----------------------

template<class T>
torch::Scalar scalar_from_r_impl_ (const SEXP x) {
  auto value = Rcpp::as<T>(x);
  torch::Scalar s_val(value);
  return s_val;
}

torch::Scalar scalar_from_r_ (SEXP x) {
  switch (TYPEOF(x)) {
  case INTSXP:
    return scalar_from_r_impl_<int>(x);
  case REALSXP:
    return scalar_from_r_impl_<double>(x);
  case LGLSXP:
    return scalar_from_r_impl_<bool>(x);
  case CHARSXP:
    Rcpp::stop("strings are not handled yet");
  default:
    Rcpp::stop("not handled");
  }
};

template<class T>
SEXP scalar_to_r_impl_ (const torch::Scalar x) {
  return Rcpp::as<T>(x);
}

SEXP scalar_to_r_ (torch::Scalar x) {
  auto a = x.to<SEXP>();
  return a;
};

// Converting scalar types to and from strings ---------------

torch::ScalarType scalar_type_from_string(std::string scalar_type) {
  if (scalar_type == "int32" | scalar_type == "int") {
    return torch::kInt;
  } else if (scalar_type == "float64" | scalar_type == "double") {
    return torch::kDouble;
  } else if (scalar_type == "float32" | scalar_type == "float") {
    return torch::kFloat;
  } else if (scalar_type == "uint8") {
    return torch::kByte;
  } else if (scalar_type == "int64" | scalar_type == "long") {
    return torch::kLong;
  }
  Rcpp::stop("ScalarType not handled yet");
}

torch::ScalarType scalar_type_from_string(Rcpp::Nullable<std::string> scalar_type) {
  if (scalar_type.isNull()) {
    return torch::ScalarType::Undefined;
  } else {
    return scalar_type_from_string(Rcpp::as<std::string>(scalar_type));
  }
}

std::string scalar_type_to_string(torch::ScalarType scalar_type) {
  if (scalar_type == torch::kInt) {
    return "int";
  } else if (scalar_type == torch::kDouble) {
    return "double";
  } else if (scalar_type == torch::kFloat) {
    return "float";
  } else if (scalar_type == torch::kLong) {
    return "long";
  }
  Rcpp::stop("Scalar Type not handled");
}

torch::Dtype dtype_from_string (std::string dtype) {
  return scalar_type_from_string(dtype);
}

std::string dtype_to_string (torch::Dtype dtype) {
  return scalar_type_to_string(dtype);
}

// Caffe Types

std::string caffe_type_to_string (caffe2::TypeMeta type) {
  std::string name = type.name();
  if (name == "unsigned char") {
    name = "uint8";
  } else if (name == "c10::Half") {
    name = "half";
  }
  return name;
}
