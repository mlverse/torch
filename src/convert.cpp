#include "torch_types.h"
#include "scalar.hpp"
#include "device.hpp"
#include "utils.hpp"

// Tensor from R code ----------------------------------------------------------

std::vector<int64_t> reverse_int_seq (int n) {
  std::vector<int64_t> l(n);
  std::iota(l.begin(), l.end(), 0);
  std::reverse(l.begin(), l.end());
  return l;
};

torch::Layout layout_from_string (std::string layout) {
  if (layout == "strided") {
    return torch::Layout::Strided;
  } else if (layout == "sparse") {
    return torch::Layout::Sparse;
  } else {
    Rcpp::stop("Layout type not implemented.");
  }
}

torch::TensorOptions tensor_options_ (Rcpp::Nullable<std::string> dtype,
                                      Rcpp::Nullable<std::string> layout,
                                      Rcpp::Nullable<std::string> device,
                                      Rcpp::Nullable<bool> requires_grad) {

  auto options = torch::TensorOptions();

  if (dtype.isNotNull()) {
    options = options.dtype(scalar_type_from_string(Rcpp::as<std::string>(dtype)));
  }

  if (layout.isNotNull()) {
    options = options.layout(layout_from_string(Rcpp::as<std::string>(layout)));
  }

  if (device.isNotNull()) {
    options = options.device(device_from_string(Rcpp::as<std::string>(device)));
  }

  if (requires_grad.isNotNull()) {
    options = options.requires_grad(Rcpp::as<bool>(requires_grad));
  }

  return options;
}

template <int RTYPE, torch::ScalarType SCALARTYPE>
torch::Tensor tensor_from_r_impl_ (const SEXP x, const std::vector<int64_t> dim) {

  Rcpp::Vector<RTYPE> vec(x);

  auto tensor = torch::from_blob(vec.begin(), dim, SCALARTYPE);

  if (dim.size() == 1) {
    // if we have a 1-dim vector contigous doesn't trigger a copy, and
    // would be unexpected.
    tensor = tensor.clone();
  }

  tensor = tensor
    .permute(reverse_int_seq(dim.size()))
    .contiguous(); // triggers a copy!

  return tensor;
};

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_from_r_ (SEXP x, std::vector<int64_t> dim,
                                          Rcpp::Nullable<std::string> dtype,
                                          Rcpp::Nullable<std::string> device,
                                          bool requires_grad = false
) {

  torch::Tensor tensor;

  if (TYPEOF(x) == INTSXP) {
    tensor = tensor_from_r_impl_<INTSXP, torch::kInt>(x, dim);
  } else if (TYPEOF(x) == REALSXP) {
    tensor = tensor_from_r_impl_<REALSXP, torch::kDouble>(x, dim);
  } else if (TYPEOF(x) == LGLSXP) {
    tensor = tensor_from_r_impl_<LGLSXP, torch::kInt32>(x, dim);
  } else {
    Rcpp::stop("R type not handled");
  };

  torch::TensorOptions options = tensor_options_(dtype, R_NilValue, device, R_NilValue);

  if (dtype.isNull()) {
    if (TYPEOF(x) == REALSXP) {
      options = options.dtype(torch::kFloat);
    } else if (TYPEOF(x) == LGLSXP) {
      options = options.dtype(torch::kByte);
    }
  }

  tensor = tensor.to(options);
  tensor = tensor.set_requires_grad(requires_grad);

  return make_tensor_ptr(tensor);
};

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_ (Rcpp::XPtr<torch::Tensor> x,
                                   Rcpp::Nullable<std::string> dtype,
                                   Rcpp::Nullable<std::string> device,
                                   bool requires_grad) {
  torch::Tensor tensor = x->clone();
  tensor = tensor.to(tensor_options_(dtype, R_NilValue, device, R_NilValue));
  tensor = tensor.set_requires_grad(requires_grad);
  return make_tensor_ptr(tensor);
}

// Tensor to R code ------------------------------------------------------------

template <int RTYPE, typename STDTYPE>
Rcpp::List as_array_tensor_impl_ (torch::Tensor x) {

  Rcpp::IntegerVector dimensions(x.ndimension());
  for (int i = 0; i < x.ndimension(); ++i) {
    dimensions[i] = x.size(i);
  }

  auto ten = x.contiguous();

  Rcpp::Vector<RTYPE> vec(ten.data<STDTYPE>(), ten.data<STDTYPE>() + ten.numel());

  return Rcpp::List::create(Rcpp::Named("vec") = vec, Rcpp::Named("dim") = dimensions);
}

// [[Rcpp::export]]
Rcpp::List as_array_tensor_ (Rcpp::XPtr<torch::Tensor> x) {

  torch::Tensor ten = *x;

  if (ten.dtype() == torch::kInt) {
    return as_array_tensor_impl_<INTSXP, int32_t>(ten);
  } else if (ten.dtype() == torch::kDouble) {
    return as_array_tensor_impl_<REALSXP, double>(ten);
  } else if (ten.dtype() == torch::kByte) {
    return as_array_tensor_impl_<LGLSXP, std::uint8_t>(ten);
  } else if (ten.dtype() == torch::kLong) {
    return as_array_tensor_impl_<INTSXP, int32_t>(ten.to(torch::kInt));
  } else if (ten.dtype() == torch::kFloat) {
    return as_array_tensor_impl_<REALSXP, double>(ten.to(torch::kDouble));
  } else if (ten.dtype() == torch::kInt16) {
    return as_array_tensor_impl_<INTSXP, int16_t>(ten.to(torch::kInt16));
  }

  Rcpp::stop("dtype not handled");
};

// [[Rcpp::export]]
void tensor_print_ (Rcpp::XPtr<torch::Tensor> x) {
  torch::Tensor ten = *x;
  Rcpp::Rcout << ten << std::endl;
};

